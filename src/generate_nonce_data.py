# %%
import argparse
from typing import Any
from random import sample
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from math import ceil
from pathlib import Path
from functools import partial
import json
import os
import multiprocessing

import tqdm
import spacy
from spacy.tokens import Token

from lib.dataset import load_custom_dataset, load_texts_from_dataset_batch
from lib.dataset import slice_dataset, skip_dataset_by_column, simple_split_to_sents
from lib.parser import extract_token_morph_features, is_content_word, is_vowel

# if spacy.prefer_gpu():
#     print("Using GPU")
# else:
#     print("Using CPU")

CPU_NUM = multiprocessing.cpu_count()
NLP = None
BATCH_SIZE = 1_000
NONCE_WORD_BANK = {}


def get_nlp():
    global NLP
    if NLP is None:
        NLP = spacy.load("en_core_web_sm")
    return NLP


# ================= Utils =================
def serialize_morph(morph_tuple) -> str:
    # morph_tuple: (pos, dep, dir, morph)
    pos, dep, dir, morph = morph_tuple
    return f"{pos}|{dep}|{dir}|{str(morph)}"


def match_nonce_words(token: Token, max_n: int) -> list[str]:
    """
    Matches a token with a list of words based on its pos and morph features

    Args:
        token (Token): A spaCy Token object.

    Returns:
        list: A list of matched words.
    """
    text, lemma, morph = extract_token_morph_features(token)
    key = serialize_morph(morph)
    candidates = NONCE_WORD_BANK.get(key, [])
    # 最大 max_n x 5 の候補単語からマーチする
    candidates = sample(candidates, min(len(candidates), max_n * 5))
    nonce_words = []
    for nonce_text, nonce_lemma in candidates:
        if nonce_text == text or nonce_lemma == lemma:
            continue
        if is_vowel(nonce_text[0]) != is_vowel(text[0]):
            continue
        if nonce_text[0].isupper() != text[0].isupper():
            continue
        nonce_words.append(nonce_text)
        if len(nonce_words) >= max_n:
            break
    return nonce_words


def generate_nonce_sentence(doc, max_n: int) -> list[str]:
    """Generates a nonce sentence by replacing tokens in the document with nonce words.
    Args:
        doc (Doc): A spaCy Doc object.
        max_n (int): The number of nonce words to generate for each token.
    returns:
        list[str]: A list of nonce words forming a sentence.
    """
    # get content words
    content_words = [token for token in doc if is_content_word(token)]

    # get nonce words for each content word
    nonce_words_per_token = []
    for token in content_words:
        candidates = match_nonce_words(token, max_n)
        if not candidates:
            # if no nonce words found, skip this sentence and return an empty list
            # make sure the nonce data is nonsensical enough
            return []

        max_n = min(len(candidates), max_n)
        candidates = sample(candidates, max_n)
        # nonce_words_per_token:
        # nonce_words for content token[0]: [n0_1, n0_2, n0_3]
        # nonce_words for content token[1]: [n1_1, n1_2, n1_3, n1_4]
        # nonce_words for content token[2]: [n2_1, n1_2]
        # ...
        nonce_words_per_token.append(candidates)

    content_indices = [t.i for t in content_words]
    ori_words = [t.text for t in doc]
    nonce_sentences = []
    for cand_i in range(max_n):
        nonce_sent_words = ori_words.copy()
        # replace content words with nonce candidates
        for cont_i, index in enumerate(content_indices):
            nonce_sent_words[index] = nonce_words_per_token[cont_i][cand_i]
        nonce_sentences.append(" ".join(nonce_sent_words).lower())

    # for combo in zip(*nonce_words_per_token):
    #     # generate nonce words to form a new sentence
    #     nonce_sent_words = ori_words.copy()
    #     for i, index in enumerate(content_indices):
    #         nonce_sent_words[index] = combo[i]
    #     nonce_sentences.append(" ".join(nonce_sent_words))
    return nonce_sentences


# ================= Blacklist =================
# Blacklist words that should not be used as nonce words
# e.g., words with multiple POS tags, stop words, etc.
# We will generate the blacklist based on the dataset
def count_pos_tags(texts, multi_process: bool, update_dict: dict | None = None) -> dict:
    """
    Counts the occurrences of each POS tag in a list of spaCy Doc objects.

    Args:
        docs (list[Doc]): A list of spaCy Doc objects.

    Returns:
        dict: A dictionary with POS tags as keys and their counts as values.
    """
    pos_counts: dict = update_dict if update_dict else {}
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["parser", "ner", "textcat", "tok2vec", "morphologizer"]
    )
    if multi_process:
        docs = nlp.pipe(texts, batch_size=512, n_process=CPU_NUM)
    else:
        docs = nlp.pipe(texts, batch_size=512)

    for doc in tqdm.tqdm(docs, total=len(texts), desc="Counting POS tags"):
        for token in doc:
            lemma = token.lemma_
            pos = token.pos_
            pos_counts.setdefault(lemma, {}).setdefault(pos, 0)
            pos_counts[lemma][pos] += 1
    return pos_counts


def generate_lemma_blacklist(
    pos_counts: dict
) -> set:
    blacklist = set()
    for lemma, pos_dict in tqdm.tqdm(pos_counts.items(), desc="Generating lemma blacklist", total=len(pos_counts)):
        total = sum(pos_dict.values())
        if max(pos_dict.values()) / total < 0.9:
            blacklist.add(lemma)
    return blacklist


def _generate_nonce_word_bank(texts, lemma_blacklist: set, multi_process, update_dict: dict | None = None) -> dict:
    """
    Extracts morphological features from a Docs object.
    """
    features = update_dict if update_dict else {}
    # need the full pipeline for sentence segmentation
    nlp = get_nlp()
    if multi_process:
        docs = nlp.pipe(texts, batch_size=64, n_process=CPU_NUM)
    else:
        docs = nlp.pipe(texts, batch_size=64)
    for doc in tqdm.tqdm(docs, total=len(texts), desc="Generating nonce words"):
        for token in doc:
            text, lemma, morph = extract_token_morph_features(token)
            if lemma in lemma_blacklist:
                continue
            morph_str = serialize_morph(morph)
            # print(morph_str)
            # given the same morph features, we want to have a list of words
            # to choose from when generating nonce words
            features.setdefault(morph_str, []).append((text, lemma))
    return {m: list(set(f)) for m, f in features.items()}


# ================= Dataset Processing =================
def map_nonce_generation(examples, multi_process):
    # need the full pipeline for sentence segmentation
    nlp = get_nlp()
    if multi_process:
        docs = nlp.pipe(examples["text"], batch_size=64, n_process=CPU_NUM)
    else:
        docs = nlp.pipe(examples["text"], batch_size=64)
    nonce = []
    for doc in docs:
        _nonce = generate_nonce_sentence(doc, 1)
        nonce.append(_nonce[0] if _nonce else "")
    examples["nonce"] = nonce
    return examples


def generate_nonce_for_dataset(
    dataset: Dataset | Any, batch_size: int, out_path: str,
    multi_process: bool,
    lemma_blacklist_path: str | Path = "",
    nonce_word_bank_path: str | Path = "",
    lemma_blacklist_generation: bool = True,
    nonce_word_bank_generation: bool = True,
    nonce_data_generation: bool = True,
):
    """
    Main function to generate nonce sentences from a list of texts.

    Args:
        texts (list[str]): A list of input texts.
        batch_size (int): The number of texts to process in each batch.
        out_path (str): The path to save the dataset with nonce sentences.
        limit (int): The maximum number of samples to process. Defaults to 0.

    Returns:
        list[str]: A list of generated nonce sentences.
    """
    # Limit the number of samples to process
    # if limit > 0:
    #     limit = min(limit, len(dataset))
    #     dataset = dataset.select(range(limit))

    batch_number = ceil(dataset.num_rows / batch_size)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {batch_size}...")

    # ========== Generate lemma blacklist ==========
    out_path_blacklist = Path(out_path) / "lemma_blacklist"
    lemma_blacklist = set()
    if lemma_blacklist_path and Path(lemma_blacklist_path).exists():
        # try to load existing lemma blacklist
        # If it exists, use it to speed up the process
        print(f"**Loading existing lemma blacklist from {lemma_blacklist_path}...")
        with open(lemma_blacklist_path, "r") as f:
            lemma_blacklist = set([line.strip() for line in f.readlines()])
        print("**done")
    elif not lemma_blacklist_generation:
        print("**Skipping lemma blacklist generation.")
    else:
        pos_counts = {}
        for i in range(batch_number):
            print(f"Generating blacklist for batch {i + 1}/{batch_number}...")
            texts = load_texts_from_dataset_batch(dataset, i, batch_size)
            pos_counts = count_pos_tags(texts, multi_process, pos_counts)
        lemma_blacklist = generate_lemma_blacklist(pos_counts)
        with open(out_path_blacklist, "w") as f:
            for lemma in lemma_blacklist:
                f.write(f"{lemma}\n")

    # ========== Generate nonce word bank ==========
    out_path_word_bank = Path(out_path) / "nonce_word_bank.json"
    global NONCE_WORD_BANK
    if nonce_word_bank_path and Path(nonce_word_bank_path).exists():
        # Load existing nonce word bank if it exists
        # This speeds up the process if the bank is already generated
        print(f"**Loading existing nonce_word_bank from {nonce_word_bank_path}...")
        with open(nonce_word_bank_path, "r") as f:
            NONCE_WORD_BANK = json.load(f)
        NONCE_WORD_BANK = {k: list(set([tuple(t) for t in v])) for k, v in NONCE_WORD_BANK.items()}
        print("**done")
    elif not nonce_word_bank_generation:
        print("**Skipping nonce word bank generation.")
    else:
        NONCE_WORD_BANK = {}
        for i in range(batch_number):
            print(f"Generating nonce bank for batch {i + 1}/{batch_number}...")
            texts = load_texts_from_dataset_batch(dataset, i, batch_size)
            NONCE_WORD_BANK = _generate_nonce_word_bank(texts, lemma_blacklist, multi_process, NONCE_WORD_BANK)
        _nonce_word_bank = {k: tuple(v) for k, v in NONCE_WORD_BANK.items()}
        json.dump(_nonce_word_bank, open(out_path_word_bank, "w"), indent=4)
        print(f"Saved nonce word bank to {out_path_word_bank}")

    # ========= Generate nonce sentences ==========
    if not nonce_data_generation:
        print("**Skipping nonce data generation.")
        return None
    print("**** Preprocessing...")
    print("**** Generating nonce sentence...")
    process_fn = partial(map_nonce_generation, multi_process=multi_process)
    dataset = dataset.map(
        process_fn,
        num_proc=os.cpu_count(),
        batch_size=batch_size,
        batched=True,
        writer_batch_size=1000,
        desc="Generating nonce sentences"
    )
    dataset = dataset.filter(lambda x: x["nonce"] != "")

    print(f"Generated {len(dataset)} samples with nonce sentences.")
    return dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str,
        help='Dataset path to load from.'
    )
    parser.add_argument(
        '--data-type', '-dt', dest='data_type', type=str, required=False, default=None,
        help=(
            'Type of the dataset to load. '
            'If not provided, the dataset will be loaded as a Hugging Face Dataset.'
        )
    )
    parser.add_argument(
        '--load-from', '-lf', dest='load_from', choices=["hf", "local"],
        help='Load dataset from Hugging Face or local path.'
    )
    parser.add_argument(
        '--start-from', '-sf', dest='start_from', type=int, default=0, required=False,
        help='Load data from line.'
    )
    parser.add_argument(
        '--limit', '-l', dest='data_limit', type=int, default=0, required=False,
        help='Limit the number of samples to process. 0 means no limit.'
    )
    parser.add_argument(
        '--lemma-blacklist', '-lb', dest='lemma_blacklist', type=str, default="",
        help='Path to existing lemma blacklist.'
    )
    parser.add_argument(
        '--nonce-word-bank', '-wb', dest='word_bank', type=str, default="",
        help='Path to existing nonce word bank.'
    )
    parser.add_argument(
        '--skip-lemma-blacklist', '-slb', dest='skip_lemma_blacklist_generation', action='store_true',
        help='Skip lemma blacklist generation.'
    )
    parser.add_argument(
        '--skip-word-bank', '-swb', dest='skip_nonce_word_bank_generation', action='store_true',
        help='Skip nonce word bank generation.'
    )
    parser.add_argument(
        '--skip-nonce-data', '-snd', dest='skip_nonce_data_generation', action='store_true',
        help='Skip nonce data generation.'
    )
    parser.add_argument(
        '--split-sents', '-ss', dest='split_sents', type=str, default="",
        help='Split data to sents before generating.'
    )
    parser.add_argument(
        '--multi-process', '-mp', dest='multi_process', action='store_true',
        help='Use multi-processing for nonce sentence generation.'
    )
    parser.add_argument(
        '--skip-key', '-sk', dest='skip_key',
        help='Skip data based on column'
    )
    parser.add_argument(
        '--skip-values', '-sv', nargs='+', dest='skip_values',
        help='Skip data based on values of the skip_key'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def main():
    args = read_args()
    print(vars(args))
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if args.multi_process:
        print("Multi process for NLP.pipe")
    else:
        print("NON-Multi process for NLP.pipe")

    # ========  Load dataset ========
    print("**** Loading dataset...")
    dataset = load_custom_dataset(
        data_name=args.data_name,
        data_type=args.data_type,
        load_from=args.load_from
    )
    print(f"Dataset loaded with {dataset.num_rows} samples.")

    # ======== Generate nonce sentences ========
    if isinstance(dataset, DatasetDict):
        dataset_dict = {}
        for key, dt in dataset.items():
            dt_limit = args.data_limit if key == "train" else int(args.data_limit * 0.1)
            start_from = args.start_from if key == "train" else int(args.start_from * 0.1) 
            dt = slice_dataset(dt, start_from, dt_limit)
            print(f"========= Processing dataset {key}... ==========")
            print(f"Dataset {key} has {dt.num_rows} samples after slicing.")
            if args.skip_key and args.skip_values:
                dt = skip_dataset_by_column(dt, args.skip_key, args.skip_values)
            if args.split_sents:
                dt = simple_split_to_sents(dt, args.split_sents, os.cpu_count(), BATCH_SIZE)
            _dataset = generate_nonce_for_dataset(
                dt,
                batch_size=BATCH_SIZE,
                out_path=out_path,
                multi_process=args.multi_process,
                lemma_blacklist_path=Path(args.lemma_blacklist),
                nonce_word_bank_path=Path(args.word_bank),
                lemma_blacklist_generation=not args.skip_lemma_blacklist_generation,
                nonce_word_bank_generation=not args.skip_nonce_word_bank_generation,
                nonce_data_generation=not args.skip_nonce_data_generation
            )
            if _dataset is not None:
                dataset_dict[key] = _dataset
        if "train" in dataset_dict:
            dataset_dict["train"].select(range(5)).to_json(Path(out_path) / "example_nonce_sent.json")
        if dataset_dict:
            print(f"Saving dataset with nonce sentences to {out_path}...")
            dataset_dict = DatasetDict(dataset_dict)
            print("Dataset structure:", dataset_dict)
            dataset_dict.save_to_disk(out_path)
    else:
        print("**** Processing dataset ...")
        dataset = slice_dataset(dataset, args.start_from, args.data_limit)
        if args.skip_key and args.skip_values:
            dataset = skip_dataset_by_column(dataset, args.skip_key, args.skip_values)
        if args.split_sents:
            dataset = simple_split_to_sents(dataset, args.split_sents, os.cpu_count(), BATCH_SIZE)
        print(f"Dataset has {dataset.num_rows} samples after slicing.")
        dataset = generate_nonce_for_dataset(
            dataset,
            batch_size=BATCH_SIZE,
            out_path=out_path,
            multi_process=args.multi_process,
            lemma_blacklist_path=Path(args.lemma_blacklist),
            nonce_word_bank_path=Path(args.word_bank),
            lemma_blacklist_generation=not args.skip_lemma_blacklist_generation,
            nonce_word_bank_generation=not args.skip_nonce_word_bank_generation,
            nonce_data_generation=not args.skip_nonce_data_generation
        )
        if dataset:
            print(f"Dataset has {dataset.num_rows} samples after generating nonce sentences.")
            print(f"Saving dataset with nonce sentences to {out_path}...")
            dataset.save_to_disk(out_path)


if __name__ == "__main__":
    main()
