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

import tqdm
import spacy
from spacy.tokens import Doc, Token
import torch

from lib.dataset import load_custom_dataset, load_texts_from_dataset_batch
from lib.parser import extract_token_morph_features, is_content_word, is_vowel

if spacy.prefer_gpu():
    print("Using GPU")
else:
    print("Using CPU")
NLP = spacy.load("en_core_web_trf", disable=["ner", "textcat", "tok2vec", "parser"])

BATCH_SIZE = 10000   # Default batch size for processing texts

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def count_pos_tags(docs, total: int, update_dict: dict | None = None) -> dict:
    """
    Counts the occurrences of each POS tag in a list of spaCy Doc objects.

    Args:
        docs (list[Doc]): A list of spaCy Doc objects.

    Returns:
        dict: A dictionary with POS tags as keys and their counts as values.
    """
    pos_counts: dict = {}
    if update_dict is not None:
        pos_counts = update_dict

    for doc in tqdm.tqdm(docs, total=total, desc="Counting POS tags"):
        for token in doc:
            lemma: str = token.lemma_
            pos: str = token.pos_
            if lemma not in pos_counts:
                pos_counts[lemma] = {}
            if pos not in pos_counts[lemma]:
                pos_counts[lemma][pos] = 0
            pos_counts[lemma][pos] += 1
    return pos_counts


def generate_nonce_words_blacklist_by_pos_tags_count(pos_counts: dict) -> list:
    # exclude the forms that appeared with a different POS
    # more than 10% of the time
    blacklist = []
    for lemma in pos_counts:
        total_count = sum(pos_counts[lemma].values())
        if max(pos_counts[lemma].values()) / total_count < 0.9:
            blacklist.append(lemma)
    return blacklist


def _generate_nonce_word_bank(docs, total: int, lemma_blacklist: list | set, update_dict: dict | None = None) -> dict:
    """
    Extracts morphological features from a Docs object.
    """
    features = {}
    if update_dict is not None:
        features = update_dict
    for doc in tqdm.tqdm(docs, total=total, desc="Generating nonce words"):
        for token in doc:
            text, lemma, morph = extract_token_morph_features(token)
            if lemma in lemma_blacklist:
                continue
            morph_str = serialize_morph(morph)
            if morph_str not in features:
                features[morph_str] = [(text, lemma)]
            elif (text, lemma) not in features[morph_str]:
                features[morph_str].append((text, lemma))
    return features


def match_nonce_words(token: Token, nonce_word_bank: dict) -> list[str]:
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
    morph_str = serialize_morph(morph)
    candidate = nonce_word_bank.get(morph_str, [])
    nonce_words = []
    for nonce_text, nonce_lemma in candidate:
        if nonce_text == text or nonce_lemma == lemma:
            continue
        if is_vowel(nonce_text[0]) != is_vowel(text[0]):
            continue
        if nonce_text[0].isupper() != text[0].isupper():
            continue
        nonce_words.append(nonce_text)
    return nonce_words


def map_process(example, nonce_word_bank):
    doc = NLP(example["text"])
    nonce = generate_nonce_sentence(doc, nonce_word_bank, 1)
    example["nonce"] = nonce[0] if nonce else ""
    return example


def serialize_morph(morph_tuple):
    # morph_tuple: (pos, dep, dir, morph)
    pos, dep, dir, morph = morph_tuple
    return f"{pos}|{dep}|{dir}|{str(morph)}"


def generate_nonce_sentence(doc: Doc, nonce_word_bank: dict, max_n: int) -> list[str]:
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
    nonce_words = []
    for ct in content_words:
        _nonce_words = match_nonce_words(ct, nonce_word_bank)
        if not _nonce_words:
            # if no nonce words found, skip this sentence and return an empty list
            # make sure the nonce data is nonsensical enough
            return []
        if len(_nonce_words) < max_n:
            max_n = len(_nonce_words)
        _nonce_words = sample(_nonce_words, max_n) if len(_nonce_words) >= max_n else _nonce_words
        nonce_words.append(_nonce_words)

    content_indices = [token.i for token in content_words]
    ori_words = [token.text for token in doc]
    nonce_sentences = []
    for _nonce_words in zip(*nonce_words):
        # generate nonce words to form a new sentence
        nonce_sent_words = ori_words.copy()
        for i, index in enumerate(content_indices):
            nonce_sent_words[index] = _nonce_words[i]
        nonce_sentences.append(" ".join(nonce_sent_words))
    return nonce_sentences


def generate_lemma_blacklist(
    dataset: Dataset | Any, batch_size: int, out_path: str, batch_number: int
) -> set:
    out_path_blacklist = Path(out_path) / "lemma_blacklist"
    if out_path_blacklist.exists():
        # try to load existing lemma blacklist
        # If it exists, use it to speed up the process
        print(f"**Loading existing lemma blacklist from {out_path_blacklist}...")
        with open(out_path_blacklist, "r") as f:
            lemma_blacklist = set([line.strip() for line in f.readlines()])
    else:
        # Generate lemma blacklist if it does not exist
        print("** Generating lemma blacklist...")
        pos_counts = {}
        for i in range(batch_number):
            # For words whose POS tags are not consistent
            # across the dataset, we will not use them as nonce words
            print(f"* Processing batch {i + 1}/{batch_number}...")
            texts = load_texts_from_dataset_batch(dataset, i, batch_size)
            docs = NLP.pipe(texts, batch_size=64)
            pos_counts = count_pos_tags(docs, batch_size, update_dict=pos_counts)

        lemma_blacklist = set(generate_nonce_words_blacklist_by_pos_tags_count(pos_counts))

        with open(out_path_blacklist, "w") as f:
            for lemma in lemma_blacklist:
                f.write(f"{lemma}\n")
        print(f"Generated lemma blacklist with {len(lemma_blacklist)} entries.")
        print(f"Saved lemma blacklist to {out_path_blacklist}")
    return lemma_blacklist


def generate_nonce_word_bank(
    dataset: Dataset | Any, batch_size: int, lemma_blacklist: list | set, out_path: str
) -> dict:
    nonce_word_bank = {}
    out_path_word_bank = Path(out_path) / "nonce_word_bank.json"

    if out_path_word_bank.exists():
        # Load existing nonce word bank if it exists
        # This speeds up the process if the bank is already generated
        print(f"**Loading existing nonce_word_bank from {out_path_word_bank}...")
        with open(out_path_word_bank, "r") as f:
            nonce_word_bank = json.load(f)
    else:
        # Sample 10% to generate nonce word bank
        bank_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)["test"]
        _batch_size = min(batch_size, len(bank_dataset))
        _batch_number = ceil(bank_dataset.num_rows / _batch_size)
        for i in range(_batch_number):
            print(f"*** Processing batch {i + 1}/{_batch_number}...")
            texts = load_texts_from_dataset_batch(bank_dataset, i, _batch_size)
            docs = NLP.pipe(texts, batch_size=64)
            nonce_word_bank = _generate_nonce_word_bank(
                docs,
                _batch_size,
                lemma_blacklist,
                update_dict=nonce_word_bank
            )
        print(f"Generated nonce word bank with {len(nonce_word_bank)} entries.")
        json.dump(nonce_word_bank, open(out_path_word_bank, "w"), indent=4)
        print(f"Saved nonce word bank to {out_path_word_bank}")
    return nonce_word_bank


def generate_nonce_for_dataset(
    dataset: Dataset | Any, batch_size: int, out_path: str, limit: int = 0
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
    if limit > 0:
        limit = min(limit, len(dataset))
        batch_size = min(batch_size, limit, len(dataset))
        dataset = dataset.select(range(limit))

    batch_number = ceil(dataset.num_rows / batch_size)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {batch_size}...")

    # ========== Generate lemma blacklist ==========
    print("**** Generating lemma blacklist...")
    lemma_blacklist = generate_lemma_blacklist(
        dataset,
        batch_size=batch_size,
        out_path=out_path,
        batch_number=batch_number
    )

    # ========== Generate nonce word bank ==========
    print("**** Generating nonce word bank...")
    nonce_word_bank = generate_nonce_word_bank(
        dataset,
        batch_size=batch_size,
        lemma_blacklist=lemma_blacklist,
        out_path=out_path
    )

    # ========= Generate nonce sentences ==========
    print("**** Generating nonce sentence...")
    process_fn = partial(map_process, nonce_word_bank=nonce_word_bank)
    dataset = dataset.map(
        process_fn,
        num_proc=1,
        writer_batch_size=10_000,
        desc="Generating nonce sentences"
    )
    dataset = dataset.filter(lambda x: x["nonce"] != "")

    print(f"Generated {len(dataset)} samples with nonce sentences.")
    return dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', '-dp', dest='data_path', type=str,
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
        '--limit', '-l', dest='data_limit', type=int, default=0, required=False,
        help='Limit the number of samples to process. 0 means no limit.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def main():
    args = read_args()
    data_limit = args.data_limit

    # ========  Load dataset ========
    print("**** Loading dataset...")
    dataset = load_custom_dataset(
        data_path=args.data_path,
        data_type=args.data_type,
        load_from=args.load_from
    )

    # ======== Generate nonce sentences ========
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if isinstance(dataset, DatasetDict):
        dataset_dict = {}
        for key, dataset in dataset.items():
            print(f"========= Processing dataset {key}... ==========")
            dataset_dict[key] = generate_nonce_for_dataset(
                dataset,
                batch_size=BATCH_SIZE,
                out_path=out_path,
                limit=data_limit if key == "train" else int(data_limit * 0.1)
            )
        if "train" in dataset_dict:
            dataset_dict["train"].select(range(5)).to_json(Path(out_path) / "example_nonce_sent.json")
        print(f"Saaving dataset with nonce sentences to {out_path}...")
        dataset_dict = DatasetDict(dataset_dict)
        dataset_dict.save_to_disk(out_path)
    else:
        print("**** Processing dataset ...")
        generate_nonce_for_dataset(
            dataset,
            batch_size=BATCH_SIZE,
            out_path=out_path,
            limit=data_limit
        ).save_to_disk(out_path)


if __name__ == "__main__":
    main()
