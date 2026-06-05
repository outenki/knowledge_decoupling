"""
Nonce data handling utilities.
"""
import multiprocessing
import json
import lmdb
import pickle
from collections import OrderedDict
from typing import Any
from math import ceil
from functools import partial
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

import spacy
from spacy.tokens import Token
import random
from datasets.arrow_dataset import Dataset

from src.lib.parser import extract_token_morph_features, is_content_word, is_vowel
from src.lib.text import safe_texts
from src.lib.text import split_text_to_sentences


CPU_NUM = min(4, multiprocessing.cpu_count())
NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 64
NONCE_WORD_BANK = {}
NONCE_WORD_BANK_SOURCE = None
NONCE_WORD_BANK_INDEX = {}
NONCE_WORD_BANK_CACHE_SIZE = 4096
random.seed(42)


class LMDBNonceWordBank:
    def __init__(self, db_path: str | Path, cache_size: int = NONCE_WORD_BANK_CACHE_SIZE):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Missing nonce word bank database: {self.db_path}")
        self._env = lmdb.open(
            str(self.db_path),
            subdir=self.db_path.is_dir(),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1024,
        )
        self._cache: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        self.cache_size = cache_size

    def get(self, morph_key: str, default: list | None = None) -> list[tuple[str, str]]:
        if morph_key in self._cache:
            self._cache.move_to_end(morph_key)
        else:
            with self._env.begin() as txn:
                payload = txn.get(morph_key.encode("utf-8"))
            self._cache[morph_key] = pickle.loads(payload) if payload is not None else []
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        if default is None:
            default = []
        return self._cache.get(morph_key, default)

    def items(self):
        with self._env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    yield key.decode("utf-8"), pickle.loads(value)

    def __len__(self) -> int:
        with self._env.begin() as txn:
            stat = txn.stat()
        return int(stat["entries"])


def _estimate_lmdb_map_size(bank: dict[str, list]) -> int:
    estimated = 0
    sample_limit = 1024
    for idx, (morph, words) in enumerate(bank.items()):
        estimated += len(morph.encode("utf-8")) + len(pickle.dumps(words, protocol=pickle.HIGHEST_PROTOCOL))
        if idx + 1 >= sample_limit:
            avg = estimated / sample_limit
            estimated = int(avg * len(bank))
            break
    if not estimated:
        estimated = 1 << 20
    return max(estimated * 2, 1 << 30)


def create_lmdb_nonce_word_bank(bank: dict[str, list], out_path: str | Path) -> None:
    out_path = Path(out_path)
    if out_path.exists():
        if out_path.is_dir():
            for child in out_path.iterdir():
                child.unlink()
        else:
            out_path.unlink()
    out_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(out_path),
        subdir=True,
        map_size=_estimate_lmdb_map_size(bank),
        readonly=False,
        meminit=False,
        map_async=True,
        writemap=True,
        lock=True,
    )
    try:
        with env.begin(write=True) as txn:
            for morph, words in tqdm(bank.items(), desc="Writing nonce bank lmdb", total=len(bank)):
                txn.put(
                    morph.encode("utf-8"),
                    pickle.dumps(words, protocol=pickle.HIGHEST_PROTOCOL),
                )
    finally:
        env.sync()
        env.close()


def save_nonce_word_bank(bank: dict[str, list], out_path: str | Path) -> None:
    out_path = Path(out_path)
    if out_path.suffix == ".lmdb" or not out_path.suffix:
        create_lmdb_nonce_word_bank(bank, out_path)
        return
    if out_path.suffix == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bank, f)
        return


def load_nonce_word_bank(bank_source: dict | str | Path):
    if isinstance(bank_source, dict):
        return bank_source

    bank_path = Path(bank_source)
    if bank_path.suffix == ".lmdb" or (bank_path.is_dir() and (bank_path / "data.mdb").exists()):
        return LMDBNonceWordBank(bank_path)
    if bank_path.suffix == ".json":
        print("Warning: loading a JSON nonce word bank will materialize the full file in memory.")
        with open(bank_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(
        f"Unsupported nonce word bank path: {bank_path}. "
        "Use a sharded bank directory or a JSON file."
    )


def serialize_morph(morph_tuple) -> str:
    # morph_tuple: (pos, tag, dep, ent, dir, morph)
    return "|".join([str(i) for i in morph_tuple])


def merge_nonce_banks(bank1: dict, bank2: dict) -> dict:
    """
    Merges two nonce word banks by concatenating the lists of words for each morph feature.
    
    Args:
        bank1 (dict): First nonce word bank {morph_key: [(text, lemma), ...], ...}
        bank2 (dict): Second nonce word bank with the same structure
    
    Returns:
        dict: Merged nonce word bank with duplicates removed
    
    Raises:
        TypeError: If inputs are not dictionaries
        ValueError: If morph values are not lists
    """
    if not isinstance(bank1, dict):
        raise TypeError(f"bank1 must be a dict, got {type(bank1)}")
    if not isinstance(bank2, dict):
        raise TypeError(f"bank2 must be a dict, got {type(bank2)}")
    
    # Use deepcopy to avoid modifying the original bank1
    merged_bank = bank1
    
    for morph, words in tqdm(bank2.items(), desc="Merging nonce banks", total=len(bank2)):
        if not isinstance(words, list):
            raise ValueError(f"morph '{morph}' has value of type {type(words)}, expected list")
        
        words = [tuple(w) for w in words]
        
        if morph in merged_bank:
            # Verify existing value is a list
            if not isinstance(merged_bank[morph], list):
                raise ValueError(f"morph '{morph}' in bank1 has type {type(merged_bank[morph])}, expected list")
            
            merged_bank[morph].extend(words)
            # Remove duplicates while preserving elements (tuples are hashable)
            # Note: set() will change order, but order is not critical for nonce banks
            merged_bank[morph] = list(set(merged_bank[morph]))
        else:
            # Deepcopy to avoid sharing references
            merged_bank[morph] = deepcopy(words)
    
    return merged_bank


def generate_nonce_word_bank(texts, multi_process) -> dict:
    """
    Extracts morphological features from a Docs object.
    """
    # features: dict mapping from morph features to a list of (text, lemma) tuples
    # if update_dict is provided, we will update the features with the new data instead of creating a new dict
    # this is useful when we want to generate nonce words for multiple datasets of texts
    # features = update_dict if update_dict else {}
    features = {}

    assert NLP is not None, "NLP should be initialized"

    # need the full pipeline for sentence segmentation
    if multi_process:
        docs = NLP.pipe(texts, batch_size=BATCH_SIZE, n_process=CPU_NUM)
    else:
        docs = NLP.pipe(texts, batch_size=BATCH_SIZE)

    for doc in docs:
        for token in doc:
            if not is_content_word(token):
                continue
            text, lemma, morph = extract_token_morph_features(token)
            morph_str = serialize_morph(morph)
            # given the same morph features, we want to have a list of words
            # to choose from when generating nonce words
            features.setdefault(morph_str, []).append((text, lemma))

    # remove duplicates in the list of words for each morph feature
    return {m: list(set(f)) for m, f in features.items()}


def match_nonce_words(token: Token, max_n: int, keep_word_identical: bool, shift: int) -> list[str]:
    """
    Matches a token with a list of words based on its pos and morph features

    Args:
        token (Token): A spaCy Token object.
        max_n (int): The maximum number of nonce sentences to return.
        keep_word_identical (bool): A word is always replaced by the same nonce word.
        shift (int): The number of positions to shift the nonce word selection.
    Returns:
        list: A list of matched words.
    """
    text, lemma, morph = extract_token_morph_features(token)
    key = serialize_morph(morph)
    candidates = NONCE_WORD_BANK.get(key, [])
    if len(candidates) <= 1:
        return []

    if keep_word_identical:
        if key not in NONCE_WORD_BANK_INDEX:
            NONCE_WORD_BANK_INDEX[key] = {
                candidate: idx for idx, candidate in enumerate(candidates)
            }
        # the start index is determined by the original word and the shift(decided by each sample),
        # so that the same word will always be replaced by the same nonce word for each sample
        start_index = NONCE_WORD_BANK_INDEX[key].get((text, lemma), 0)
        start_index = (start_index + shift) % len(candidates)
        start_index += 1
    else:
        start_index = random.randrange(len(candidates))

    nonce_words = []
    for i in range(len(candidates)):
        nonce_text, nonce_lemma = candidates[(start_index + i) % len(candidates)]
        if not nonce_text.strip().isalpha():
            # skip if the candidate word is not purely alphabetic to maintain some orthographic similarity
            continue
        if nonce_text == text or nonce_lemma == lemma:
            # skip if the candidate word is the same as the original word in text or lemma form
            continue
        if is_vowel(nonce_text[0]) != is_vowel(text[0]):
            # make sure the nonce word has the same vowel/consonant pattern as the original word to maintain some phonetic similarity
            continue
        if nonce_text[0].isupper() != text[0].isupper():
            # make sure the nonce word has the same capitalization pattern as the original word to maintain some orthographic similarity
            continue
        nonce_words.append(nonce_text)
        if len(nonce_words) >= max_n:
            break
    return nonce_words


def generate_nonce_sentence(doc, max_n: int, keep_word_identical: bool, ne_only: bool = False) -> list:
    """Generates a nonce sentence by replacing tokens in the document with nonce words.
    Args:
        doc (Doc): A spaCy Doc object.
        max_n (int): The number of nonce words to generate for each token.
        keep_word_identical (bool): A word is always replaced by the same nonce word.
        ne_only (bool): Only replace named entities.
    returns:
        list[str]: A list of nonce words forming a sentence.
    """
    # get content words
    content_words = [token for token in doc if is_content_word(token)]
    if ne_only:
        content_words = [token for token in content_words if token.ent_type_]
    if not content_words:
        return []

    # get nonce words for each content word
    nonce_words_per_token = []
    matched_content_word_num = 0
    for token in content_words:
        candidates = match_nonce_words(token, max_n, keep_word_identical, len(doc))
        if not candidates:
            # if no nonce words found, skip this sentence and return an empty list
            # make sure the nonce data is nonsensical enough
            candidates = [token.text]
        else:
            matched_content_word_num += 1

        candidates = random.sample(candidates, max_n)
        # nonce_words_per_token:
        # nonce_words for content token[0]: [n0_1, n0_2, n0_3]
        # nonce_words for content token[1]: [n1_1, n1_2, n1_3, n1_4]
        # nonce_words for content token[2]: [n2_1, n1_2]
        # ...
        nonce_words_per_token.append(candidates)
    assert len(nonce_words_per_token) == len(content_words)

    content_indices = [t.i for t in content_words]
    ori_words = [t.text_with_ws for t in doc]
    # nonce_words = random.sample(list(itertools.product(*nonce_words_per_token)), k=max_n)
    nonce_words = list(zip(*nonce_words_per_token))[:max_n]
    nonce_sentences = []
    for cand in nonce_words:
        nonce_sent_words = ori_words.copy()
        # replace content words with nonce candidates
        for cont_i, index in enumerate(content_indices):
            # the index of the cont_i th content word is index
            # we want to replace the content word at index with the cont_i th candidate nonce word
            if nonce_sent_words[index].endswith(" "):
                nonce_sent_words[index] = cand[cont_i] + " "
            else:
                nonce_sent_words[index] = cand[cont_i]
        nonce_sent = "".join(nonce_sent_words)
        nonce_sentences.append({
            "ori_text": doc.text,
            "nonce_text": nonce_sent,
            "matched_content_word_num": matched_content_word_num,
            "total_content_word_num": len(content_words)
        })

    return nonce_sentences


def generate_nonce_for_examples(examples, multi_process: bool, max_n: int, keep_word_identical: bool, ne_only: bool = False):
    assert NLP is not None, "NLP should be initialized"
    texts = examples["text"]
    sents = []
    for text in texts:
        sents.extend(split_text_to_sentences(text))

    if multi_process:
        docs = NLP.pipe(safe_texts(sents, NLP.max_length), batch_size=BATCH_SIZE, n_process=CPU_NUM)
    else:
        docs = NLP.pipe(safe_texts(sents, NLP.max_length), batch_size=BATCH_SIZE)
    
    ori_texts = []
    nonce_texts = []
    matched_content_word_nums = []
    total_content_word_nums = []
    for doc in docs:
        nonce_sentences = generate_nonce_sentence(doc, max_n, keep_word_identical, ne_only)
        for item in nonce_sentences:
            ori_texts.append(item["ori_text"])
            nonce_texts.append(item["nonce_text"])
            matched_content_word_nums.append(item["matched_content_word_num"])
            total_content_word_nums.append(item["total_content_word_num"])

    return {
        "text": ori_texts,
        "nonce": nonce_texts,
        "matched_content_word_num": matched_content_word_nums,
        "total_content_word_num": total_content_word_nums,
    }


def generate_nonce_for_dataset(
    dataset: Dataset | Any,
    multi_process: bool,
    max_n: int,
    keep_word_identical: bool,
    nonce_word_bank: dict | str | Path,
    ne_only: bool = False
):
    batch_number = ceil(dataset.num_rows / BATCH_SIZE)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {BATCH_SIZE}...")

    global NONCE_WORD_BANK, NONCE_WORD_BANK_SOURCE, NONCE_WORD_BANK_INDEX
    if NONCE_WORD_BANK_SOURCE != nonce_word_bank:
        NONCE_WORD_BANK = load_nonce_word_bank(nonce_word_bank)
        NONCE_WORD_BANK_SOURCE = nonce_word_bank
        NONCE_WORD_BANK_INDEX = {}
    print("**** Preprocessing...")
    print("**** Generating nonce sentence...")

    process_fn = partial(
        generate_nonce_for_examples,
        multi_process=multi_process,
        max_n=max_n,
        keep_word_identical=keep_word_identical,
        ne_only=ne_only
    )
    
    dataset = dataset.map(
        process_fn,
        num_proc=1,
        batch_size=BATCH_SIZE,
        batched=True,
        remove_columns=dataset.column_names,
        writer_batch_size=1000,
        desc="Generating nonce sentences",
        load_from_cache_file=False
    )
    return dataset


def clean_nonce_word_bank(
    bank: dict | LMDBNonceWordBank,
    out_path: str | Path | None = None,
) -> dict:
    """
    Cleans a nonce word bank by removing morph features that have too few candidate words.
    This can help improve the quality of generated nonce sentences by ensuring more variability in nonce word selection.
    """
    cleaned_bank = {
        morph: words
        for morph, words in tqdm(bank.items(), desc="Cleaning nonce word bank", total=len(bank))
        if len(words) > 1
    }
    removed_count = len(bank) - len(cleaned_bank)
    print(f"Cleaned nonce word bank: removed {removed_count} morph features with <= 1 candidate words.")

    if out_path is not None:
        print(f"Saving cleaned nonce word bank to {out_path}...")
        save_nonce_word_bank(cleaned_bank, out_path)
        print(f"Saved cleaned nonce word bank to {out_path}.")

    return cleaned_bank
