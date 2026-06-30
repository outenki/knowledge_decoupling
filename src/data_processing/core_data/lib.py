"""
Nonce data handling utilities.
"""
import multiprocessing
from typing import Any
from math import ceil
from functools import partial
from pathlib import Path

import spacy
from spacy.tokens import Token
import random
from datasets.arrow_dataset import Dataset

from src.lib.parser import is_content_word
from src.lib.text import safe_texts


CPU_NUM = min(4, multiprocessing.cpu_count())
NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 64
AOA = {}
random.seed(42)

def generate_core_sentence(sent, doc_id: int, replace_ne: bool, ne_ids: dict[str, str]) -> tuple:
    """Generate a core sentence by replacing named entities with placeholders."""
    words: list[str] = []
    ne_num = 0
    content_word_num = 0

    for token in sent:
        if not is_content_word(token):
            words.append(token.text_with_ws)
            continue
        content_word_num += 1
        token_text = token.text
        token_lower = token_text.lower()
        token_lemma = token.lemma_.lower()

        # Replace named entities.
        if replace_ne and token.ent_type_:
            if token_lower not in ne_ids:
                ne_ids[token_lower] = f"{doc_id}_{len(ne_ids)}"

            word = f"{token.ent_type_.upper()}_{ne_ids[token_lower]}"
            if token.whitespace_:
                word += token.whitespace_

            words.append(word)
            ne_num += 1
            continue

        # Reject words outside AOA.
        if AOA and token_lower not in AOA and token_lemma not in AOA:
            print("\n")
            print(f"===sent({token_lower})===\n", sent.text)
            print("\n")
            return "", 0, 0

        words.append(token.text_with_ws)

    text = "".join(words)
    return text, content_word_num, ne_num


def generate_core_doc(doc, doc_id: int, replace_ne: bool) -> str | None:
    ne_ids: dict[str, str] = {}
    ne_num = 0
    content_word_num = 0
    texts = []

    for sent in doc.sents:
        t, cn, nn = generate_core_sentence(sent, doc_id, replace_ne, ne_ids)
        content_word_num += cn
        ne_num += nn
        texts.append(t)

    text = "".join(texts)
    if ne_num == content_word_num:
        return ""
    if text:
        return text
    else:
        return ""
        


def generate_core_for_examples(examples, replace_ne: bool, multi_process: bool):
    assert NLP is not None, "NLP should be initialized"
    texts = examples["text"]
    # sents = []
    # for text in texts:
    #     sents.extend(split_text_to_sentences(text))
    # sents = examples["text"]

    if multi_process:
        docs = NLP.pipe(safe_texts(texts, NLP.max_length), batch_size=BATCH_SIZE, n_process=CPU_NUM)
    else:
        docs = NLP.pipe(safe_texts(texts, NLP.max_length), batch_size=BATCH_SIZE)
    
    ori_texts = []
    core_texts = []
    for d_id, doc in enumerate(docs):
        core_sentence = generate_core_doc(doc, doc_id=d_id, replace_ne=replace_ne)
        if core_sentence:
            ori_texts.append(doc.text)
            core_texts.append(core_sentence)

    return {
        "text": ori_texts,
        "core": core_texts,
    }


def generate_core_dataset(
    dataset: Dataset | Any,
    replace_ne: bool,
    aoa: dict | None,
    multi_process: bool,
):
    global AOA
    AOA = aoa
    batch_number = ceil(dataset.num_rows / BATCH_SIZE)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {BATCH_SIZE}...")

    print("**** Preprocessing...")
    process_fn = partial(generate_core_for_examples, replace_ne=replace_ne, multi_process=multi_process)
    dataset = dataset.map(
        process_fn,
        num_proc=4,
        batch_size=BATCH_SIZE,
        batched=True,
        remove_columns=dataset.column_names,
        writer_batch_size=1000,
        desc="Generating core sentences",
        load_from_cache_file=False,
    )
    print(dataset.num_rows)
    return dataset
