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

import tqdm
import spacy
from spacy.tokens import Token

from lib.dataset import load_custom_dataset, load_texts_from_dataset_batch, slice_dataset
from lib.parser import extract_token_morph_features, is_content_word, is_vowel

# if spacy.prefer_gpu():
#     print("Using GPU")
# else:
#     print("Using CPU")

NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 100_000


# ================= Utils =================
def serialize_morph(morph_tuple) -> str:
    # morph_tuple: (pos, dep, dir, morph)
    pos, dep, dir, morph = morph_tuple
    return f"{pos}|{dep}|{dir}|{str(morph)}"


def match_nonce_words(token: Token, nonce_word_bank: dict, max_n: int) -> list[str]:
    """
    Matches a token with a list of words based on its pos and morph features

    Args:
        token (Token): A spaCy Token object.
        nonce_word_bank (list): A dict of words with morphological features
            as keys and (text, lemma) as values.

    Returns:
        list: A list of matched words.
    """
    text, lemma, morph = extract_token_morph_features(token)
    key = serialize_morph(morph)
    candidates = list(nonce_word_bank.get(key, []))
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


def generate_nonce_sentence(doc, nonce_word_bank: dict, max_n: int) -> list[str]:
    """Generates a nonce sentence by replacing tokens in the document with nonce words.
    Args:
        doc (Doc): A spaCy Doc object.
        nonce_word_bank (dict): A dict of nonce words with morphological features as keys.
        n (int): The number of nonce words to generate for each token.
    returns:
        list[str]: A list of nonce words forming a sentence.
    """
    # get content words
    content_words = [token for token in doc if is_content_word(token)]

    # get nonce words for each content word
    nonce_words_per_token = []
    for token in content_words:
        candidates = match_nonce_words(token, nonce_word_bank, max_n)
        if not candidates:
            # if no nonce words found, skip this sentence and return an empty list
            # make sure the nonce data is nonsensical enough
            return []

        max_n = min(len(candidates), max_n)
        candidates = sample(candidates, max_n)
        nonce_words_per_token.append(candidates)

    content_indices = [t.i for t in content_words]
    ori_words = [t.text for t in doc]
    nonce_sentences = []
    for combo in zip(*nonce_words_per_token):
        # generate nonce words to form a new sentence
        nonce_sent_words = ori_words.copy()
        for i, index in enumerate(content_indices):
            nonce_sent_words[index] = combo[i]
        nonce_sentences.append(" ".join(nonce_sent_words))
    return nonce_sentences


# ================= Blacklist =================
# Blacklist words that should not be used as nonce words
# e.g., words with multiple POS tags, stop words, etc.
# We will generate the blacklist based on the dataset
def count_pos_tags(texts, update_dict: dict | None = None) -> dict:
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


def _generate_nonce_word_bank(texts, lemma_blacklist: set, update_dict: dict | None = None) -> dict:
    """
    Extracts morphological features from a Docs object.
    """
    features = update_dict if update_dict else {}
    # need the full pipeline for sentence segmentation
    docs = NLP.pipe(texts, batch_size=64)
    for doc in tqdm.tqdm(docs, total=len(texts), desc="Generating nonce words"):
        for token in doc:
            text, lemma, morph = extract_token_morph_features(token)
            if lemma in lemma_blacklist:
                continue
            morph_str = serialize_morph(morph)
            # print(morph_str)
            # given the same morph features, we want to have a list of words
            # to choose from when generating nonce words
            features.setdefault(morph_str, set()).add((text, lemma))
    return features


# ================= Dataset Processing =================
def map_process(examples, nonce_word_bank):
    # need the full pipeline for sentence segmentation
    print("NLP pipe processing...")
    docs = NLP.pipe(examples["text"], batch_size=64)
    nonce = []
    for doc in docs:
        _nonce = generate_nonce_sentence(doc, nonce_word_bank, 1)
        nonce.append(_nonce[0] if _nonce else "")
    examples["nonce"] = nonce
    return examples


def generate_nonce_for_dataset(
    dataset: Dataset | Any, batch_size: int, out_path: str,
    lemma_blacklist_path: str = "",
    nonce_word_bank_path: str = "",
    lemma_blacklist_generation: bool = True,
    nonce_word_bank_generation: bool = True,
    nonce_data_generation: bool = True
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
    if lemma_blacklist_path and lemma_blacklist_path.exists():
        # try to load existing lemma blacklist
        # If it exists, use it to speed up the process
        print(f"**Loading existing lemma blacklist from {lemma_blacklist_path}...")
        with open(lemma_blacklist_path, "r") as f:
            lemma_blacklist = set([line.strip() for line in f.readlines()])
        print(f"**done")
    elif not lemma_blacklist_generation:
        print("**Skipping lemma blacklist generation.")
    else:
        pos_counts = {}
        for i in range(batch_number):
            print(f"Generating blacklist for batch {i + 1}/{batch_number}...")
            texts = load_texts_from_dataset_batch(dataset, i, batch_size)
            pos_counts = count_pos_tags(texts, pos_counts)
        lemma_blacklist = generate_lemma_blacklist(pos_counts)
        with open(out_path_blacklist, "w") as f:
            for lemma in lemma_blacklist:
                f.write(f"{lemma}\n")

    # ========== Generate nonce word bank ==========
    out_path_word_bank = Path(out_path) / "nonce_word_bank.json"
    if nonce_word_bank_path and nonce_word_bank_path.exists():
        # Load existing nonce word bank if it exists
        # This speeds up the process if the bank is already generated
        print(f"**Loading existing nonce_word_bank from {nonce_word_bank_path}...")
        with open(nonce_word_bank_path, "r") as f:
            nonce_word_bank = json.load(f)
        nonce_word_bank = {k: set([tuple(t) for t in v]) for k, v in nonce_word_bank.items()}
        print(f"**done")
    elif not nonce_word_bank_generation:
        print("**Skipping nonce word bank generation.")
    else:
        nonce_word_bank = {}
        for i in range(batch_number):
            print(f"Generating nonce bank for batch {i + 1}/{batch_number}...")
            texts = load_texts_from_dataset_batch(dataset, i, batch_size)
            nonce_word_bank = _generate_nonce_word_bank(texts, lemma_blacklist, nonce_word_bank)
        _nonce_word_bank = {k: tuple(v) for k, v in nonce_word_bank.items()}
        json.dump(_nonce_word_bank, open(out_path_word_bank, "w"), indent=4)
        print(f"Saved nonce word bank to {out_path_word_bank}")

    # ========= Generate nonce sentences ==========
    if not nonce_data_generation:
        print("**Skipping nonce data generation.")
        return None
    print("**** Generating nonce sentence...")
    process_fn = partial(map_process, nonce_word_bank=nonce_word_bank)
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
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def main():
    args = read_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)

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
            _dataset = generate_nonce_for_dataset(
                dt,
                batch_size=BATCH_SIZE,
                out_path=out_path,
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
        print(f"Dataset has {dataset.num_rows} samples after slicing.")
        dataset = generate_nonce_for_dataset(
            dataset,
            batch_size=BATCH_SIZE,
            out_path=out_path,
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
