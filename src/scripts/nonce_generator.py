# %%
from typing import Literal
from random import sample
from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_from_disk
from math import ceil
from pathlib import Path
from functools import partial

import tqdm
import spacy
from spacy.tokens import Doc, Token
import torch

spacy.require_gpu()
NLP = spacy.load("en_core_web_trf", disable=["ner", "textcat", "tok2vec", "parser"])

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def load_texts_from_dataset(dataset: Dataset, batch_idx: int, batch_size: int) -> list[str]:
    """
    Loads texts from a Hugging Face Dataset object.

    Args:
        dataset (Dataset): A Hugging Face Dataset object.

    Returns:
        list[str]: A list of texts from the dataset.
    """
    batch_size = min(len(dataset), batch_size)
    rng = range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
    if isinstance(dataset, Dataset):
        return dataset.select(rng)["text"]
    else:
        raise ValueError("Unsupported dataset type. Please provide a Hugging Face Dataset object.")


def is_content_word(token: Token) -> bool:
    return token.pos_ in (
        "ADJ", "NOUN", "VERB", "PROPN", "NUM", "ADV"
    )


def is_vowel(c: str) -> bool:
    return c.lower() in ("a", "o", "u", "e", "i", "è")


def dep_dir(token) -> Literal["left", "right", "root"]:
    """
    Returns the direction of the dependency relation for a given token.

    Args:
        token (Token): A spaCy Token object.

    Returns:
        str: The direction of the dependency relation ('left', 'right', or 'root').
    """
    if token.dep_ == "ROOT":
        return "root"
    elif token.head == token:
        return "root"
    elif token.head.i < token.i:
        return "right"
    else:
        return "left"


def extract_token_morph_features(token: Token) -> tuple:
    return (
        token.text, token.lemma_,
        (token.pos_, token.dep_, dep_dir(token), token.morph)
    )


def count_pos_tags(docs: list[Doc], total: int, update_dict: dict = None) -> dict:
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

    for doc in tqdm.tqdm(docs, total=total):
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


def generate_nonce_word_bank(docs: list[Doc], total: int, lemma_blacklist: list, update_dict: dict = None) -> dict:
    """
    Extracts morphological features from a Docs object.
    """
    features = {}
    if update_dict is not None:
        features = update_dict
    for doc in tqdm.tqdm(docs, total=total):
        for token in doc:
            text, lemma, morph = extract_token_morph_features(token)
            if lemma in lemma_blacklist:
                continue
            morph_str = serialize_morph(morph)
            if morph_str not in features:
                features[morph_str] = [(text, lemma)]
            else:
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
    example["nonce_sentence"] = nonce[0] if nonce else ""
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

    content_indexes = [token.i for token in content_words]
    ori_words = [token.text for token in doc]
    nonce_sentences = []
    for _nonce_words in zip(*nonce_words):
        # generate nonce words to form a new sentence
        nonce_sent_words = ori_words.copy()
        for i, index in enumerate(content_indexes):
            nonce_sent_words[index] = _nonce_words[i]
        nonce_sentences.append(" ".join(nonce_sent_words))
    return nonce_sentences


def generate_nonce_for_dataset(
    dataset: Dataset, output_path: str, batch_size: int,
    out_path: str, limit: int = None
):
    """
    Main function to generate nonce sentences from a list of texts.

    Args:
        texts (list[str]): A list of input texts.
        batch_size (int): The number of texts to process in each batch.
        out_path (str): The path to save the dataset with nonce sentences.
        limit (int, optional): The maximum number of samples to process. Defaults to None.

    Returns:
        list[str]: A list of generated nonce sentences.
    """
    if limit is not None:
        batch_size = min(batch_size, limit)
        dataset = dataset.select(range(limit))
    batch_number = ceil(dataset.num_rows / batch_size)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {batch_size}...")

    # try to load existing lemma blacklist
    if (Path(out_path) / "lemma_blacklist").exists():
        print(f"**Loading existing lemma blacklist from {out_path}...")
        with open(Path(out_path) / "lemma_blacklist", "r") as f:
            lemma_blacklist = [line.strip() for line in f.readlines()]
    else:
        print("\n\n** Generating lemma blacklist...")
        pos_counts = {}
        for i in range(batch_number):
            print(f"* Processing batch {i + 1}/{batch_number}...")
            texts = load_texts_from_dataset(dataset, i, batch_size)
            docs = NLP.pipe(texts, batch_size=64)
            pos_counts = count_pos_tags(docs, batch_size, update_dict=pos_counts)
        lemma_blacklist = generate_nonce_words_blacklist_by_pos_tags_count(pos_counts)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        with open(Path(out_path) / "lemma_blacklist", "w") as f:
            for lemma in lemma_blacklist:
                f.write(f"{lemma}\n")

    print("\n\n**** Generating nonce word bank...")
    nonce_word_bank = {}
    # Sample 1% to generate nonce word bank
    bank_dataset = dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)["test"]
    _batch_size = min(batch_size, len(bank_dataset))
    _batch_number = ceil(bank_dataset.num_rows / _batch_size)
    print(f"Processing {bank_dataset.num_rows} samples in {_batch_number} batches of size {_batch_size}...")
    for i in range(_batch_number):
        print(f"*** Processing batch {i + 1}/{_batch_number}...")
        texts = load_texts_from_dataset(bank_dataset, i, _batch_size)
        docs = NLP.pipe(texts, batch_size=64)
        nonce_word_bank = generate_nonce_word_bank(
            docs,
            _batch_size,
            lemma_blacklist,
            update_dict=nonce_word_bank
        )
    print(f"Generated nonce word bank with {len(nonce_word_bank)} entries.")

    print("\n\n**** Generating nonce sentence...")
    process_fn = partial(map_process, nonce_word_bank=nonce_word_bank)
    dataset = dataset.map(
        process_fn,
        batched=False,
        num_proc=1,
        writer_batch_size=10_000,
        desc="Generating nonce sentences"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "dataset_with_nonce_sentences"
    print(f"\n\n**** Saving dataset with nonce sentences to {out_file}...")
    dataset.save_to_disk(out_file)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', '-dp', dest='data_path', type=str,
        help='Dataset path to load from.'
    )
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str,
        help= 'Dataset name to load from.'
    )
    parser.add_argument(
        '--data-type', '-dt', dest='data_type', type=str, required=False, default=None,
        help=(
            'Type of the dataset to load.'
            'If not provided, the dataset will be loaded as a Hugging Face Dataset.'
        )
    )
    parser.add_argument(
        '--load-from', '-lf', dest='load_from', choices=["hf", "local"],
        help='Load dataset from Hugging Face or local path.'
    )
    parser.add_argument(
        '--limit', '-l', dest='data_limit', type=int,
        required=False, default=None,
        help='Limit the number of samples to process.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    args = parser.parse_args()
    print("\n\n**** Loading dataset...")
    if args.load_from == "local" and args.data_type is None:
        print(f"Loading dataset {args.data_path} / {args.data_name} from local disk...")
        data_path = Path(args.data_path) / args.data_name
        dataset = load_from_disk(data_path)
    elif args.load_from == "local" and args.data_type is not None:
        print(f"Loading dataset {args.data_path} / {args.data_name} from local disk with type {args.data_type}...")
        data_path = Path(args.data_path) / args.data_name
        dataset: Dataset = load_dataset(args.data_type, data_files=data_path)
    elif args.load_from == "hf":
        print(f"Loading dataset {args.data_path} / {args.data_name} from Hugging Face...")
        print(f"load_dataset('{args.data_path}', '{args.data_name}')")
        dataset: Dataset = load_dataset(args.data_path, args.data_name)
    else:
        raise ValueError("Invalid load_from option.")

    out_path = Path(args.out_path)
    if type(dataset) is DatasetDict:
        for key, dataset in dataset.items():
            print(f"\n\n**** Processing dataset {key}...")
            out_path = Path(out_path) / key
            generate_nonce_for_dataset(
                dataset,
                out_path,
                batch_size=1000,
                out_path=out_path,
                limit=args.data_limit
            )
    else:
        print("\n\n**** Processing dataset ...")
        generate_nonce_for_dataset(
            dataset,
            out_path,
            batch_size=1000,
            out_path=out_path,
            limit=args.data_limit
        )


if __name__ == "__main__":
    main()
