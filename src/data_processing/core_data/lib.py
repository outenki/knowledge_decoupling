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
AOA = {}
random.seed(42)


def generate_core_sentence(doc, d_id, replace_ne: bool, aoa_threshold: int) -> str | None:
    """Generates a nonce sentence by replacing tokens in the document with nonce words.
    Args:
        doc (Doc): A spaCy Doc object.
        max_n (int): The number of nonce words to generate for each token.
        keep_word_identical (bool): A word is always replaced by the same nonce word.
        ne_only (bool): Only replace named entities.
    returns:
        list[str]: A list of nonce words forming a sentence.
    """
    words = []
    ne_ids = {}
    for token in doc:
        if not is_content_word(token):
            words.append(token.text_with_ws)
            continue
        if replace_ne and token.ent_type_:
            # replace named entities
            ne_id = ne_ids.get(token.text.lower(), f"{d_id}_{len(ne_ids)}")
            word = token.ent_type_.upper() + "_" + ne_id
            if token.text_with_ws != token.text:
                word += " "
            words.append(word)
            continue
        if aoa_threshold > 0 and AOA.get(token.text.lower().trip, 100) < aoa_threshold:
            words.append(token.text_with_ws)
            continue
        else:
            return None
    return "".join(words)


def generate_core_for_examples(examples, multi_process: bool):
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
    core_texts = []
    for d_id, doc in enumerate(docs):
        core_sentence = generate_core_sentence(doc=doc, d_id=d_id, replace_ne=True, aoa_threshold = 0)
        if core_sentence:
            ori_texts.append(doc.text)
            core_texts.append(core_sentence)

    return {
        "text": ori_texts,
        "nonce": core_texts,
    }


def generate_core_dataset(
    dataset: Dataset | Any,
    multi_process: bool,
):
    batch_number = ceil(dataset.num_rows / BATCH_SIZE)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {BATCH_SIZE}...")

    print("**** Preprocessing...")
    print("**** Generating nonce sentence...")

    process_fn = partial(generate_core_for_examples, multi_process=multi_process)
    
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
