"""
Nonce data handling utilities.
"""
import multiprocessing
from typing import Any
from math import ceil
from functools import partial
import pandas as pd
import random

import spacy
from datasets.arrow_dataset import Dataset

from src.lib.parser import is_content_word
from src.lib.text import safe_texts


CPU_NUM = min(4, multiprocessing.cpu_count())
NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 64
AOA = {}
random.seed(42)


def load_aoa(csv: str, aoa_threshold) -> dict:
    aoa = {}
    aoa_csv = pd.read_csv(csv, usecols=["Word", "Alternative.spelling", "AoA_Kup_lem"])
    for word, alt, age in aoa_csv.itertuples(index=False, name=None):
        aoa[word] = age
        if alt not in aoa:
            aoa[alt] = age
    print(f"Loaded AoA vocabulary with {len(aoa)} entries.")

    if aoa_threshold > 0:
        aoa = {k: v for k, v in aoa.items() if v <= aoa_threshold}
        print(f"AOA threshold {aoa_threshold} kept {len(aoa)} entries.")
    else:
        print("AOA threshold is 0, so all loaded AoA entries are kept.")
    return aoa


def _new_core_word(prefix: str, core_words_for_prefix: list[str]) -> str:
    # different prefix can have the same id
    if len(core_words_for_prefix) > 1000:
        raise ValueError(f"The number of ids for {prefix} exceeded 1000.")
    core_words = tuple(core_words_for_prefix)
    for _ in range(10):
        word_id = random.randint(0, 1000)
        new_core_word = f"{prefix}_{word_id}"
        if new_core_word not in core_words:
            return new_core_word
    raise ValueError("Tried for 10 times but failed to assign new id to core word.")
    

def generate_core_sentence(sent, doc_id: int, replace_ne: bool, core_words_map: dict[str, str]) -> tuple:
    assert len(AOA) > 0
    """Generate a core sentence by replacing named entities with placeholders."""
    words: list[str] = []
    rp_ne_num = 0
    rp_unk_num = 0
    content_word_num = 0

    for token in sent:
        if not is_content_word(token):
            words.append(token.text_with_ws)
            continue
        if not token.text.isascii():
            # skip unicode chars
            continue
        content_word_num += 1
        token_text = token.text
        token_lower = token_text.lower()
        token_lemma = token.lemma_.lower()

        # Replace named entities.
        if replace_ne and token.ent_type_:
            prefix = token.ent_type_.upper() + "_" + str(doc_id)
            if token_text not in core_words_map:
                core_words_for_prefix = [v for v in core_words_map if v.startswith(prefix)]
                core_words_map[token_text] = _new_core_word(prefix, core_words_for_prefix)
            word = core_words_map[token_text]
            word = f"<{word}>"
            if token.whitespace_:
                word += token.whitespace_

            words.append(word)
            rp_ne_num += 1
            continue

        # Reject words outside AOA.
        if AOA and token_lower not in AOA and token_lemma not in AOA:
            prefix = f"UNK_{token.tag_.upper()}_{doc_id}"
            if token_text not in core_words_map:
                core_words_for_prefix = [v for v in core_words_map if v.startswith(prefix)]
                core_words_map[token_text] = _new_core_word(prefix, core_words_for_prefix)
            word = core_words_map[token_text]
            word = f"<{word}>"
            if token.whitespace_:
                word += token.whitespace_
            words.append(word)
            rp_unk_num += 1
            continue

        words.append(token.text_with_ws)

    text = "".join(words)
    return text, content_word_num, rp_ne_num, rp_unk_num


def generate_core_doc(doc, doc_id: int, replace_ne: bool) -> tuple:
    core_words_map: dict[str, str] = {}
    rp_ne_num = 0
    rp_unk_num = 0
    content_word_num = 0
    texts = []

    for sent in doc.sents:
        t, cn, nn, un= generate_core_sentence(sent, doc_id, replace_ne, core_words_map)
        content_word_num += cn
        rp_ne_num += nn
        rp_unk_num += un
        texts.append(t)

    text = "".join(texts)
    return text, content_word_num, rp_ne_num, rp_unk_num


def generate_core_for_qa(doc_id, question: str, answer: str, replace_ne, aoa: dict) -> tuple:
    global AOA
    if aoa and len(aoa) > 0:
        AOA = aoa
    core_q , core_a = question.strip(), answer.strip()
    if core_q:
        doc_q = NLP(question.strip())
        core_q = generate_core_doc(doc_q, doc_id, replace_ne)[0]
    if core_a:
        doc_qa = NLP(question.strip() + " " + answer.strip())
        core_qa = generate_core_doc(doc_qa, doc_id, replace_ne)[0]
        core_a = " ".join(core_qa.split()[len(core_q.split()):])
    return core_q, core_a


def generate_core_for_texts(texts: list[str], replace_ne: bool, multi_process: bool, lower_text: bool) -> dict:
    assert NLP is not None, "NLP should be initialized"
    if lower_text:
        texts = [t.lower() for t in texts]
    if multi_process:
        docs = NLP.pipe(safe_texts(texts, NLP.max_length), batch_size=BATCH_SIZE, n_process=CPU_NUM)
    else:
        docs = NLP.pipe(safe_texts(texts, NLP.max_length), batch_size=BATCH_SIZE)
    ori_texts = []
    core_texts = []
    content_words_num = []
    replaced_ne_num = []
    replaced_unk_num = []
    for d_id, doc in enumerate(docs):
        core_sentence , cn, nn, un= generate_core_doc(doc, doc_id=d_id, replace_ne=replace_ne)
        if core_sentence:
            ori_texts.append(doc.text)
            core_texts.append(core_sentence)
            content_words_num.append(cn)
            replaced_ne_num.append(nn)
            replaced_unk_num.append(un)
    return {
    "text": ori_texts,
    "core": core_texts,
    "content_words_num": content_words_num,
    "replaced_ne_num": replaced_ne_num,
    "replaced_unk_num": replaced_unk_num,
}


def generate_core_for_examples(examples, replace_ne: bool, multi_process: bool, lower_text: bool, column_name: str = "text") -> dict:
    texts = examples[column_name]
    # sents = []
    # for text in texts:
    #     sents.extend(split_text_to_sentences(text))
    # sents = examples["text"]
    return generate_core_for_texts(texts, replace_ne=replace_ne, multi_process=multi_process, lower_text=lower_text)


def generate_core_dataset(
    dataset: Dataset | Any,
    column_name: str,
    replace_ne: bool,
    aoa: dict | None,
    multi_process: bool,
    lower_text: bool
):
    global AOA
    AOA = aoa
    batch_number = ceil(dataset.num_rows / BATCH_SIZE)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {BATCH_SIZE}...")

    print("**** Preprocessing...")
    process_fn = partial(
        generate_core_for_examples,
        replace_ne=replace_ne,
        multi_process=multi_process,
        column_name=column_name,
        lower_text=lower_text
    )
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


def _replace_columns_with_core_data(examples, column_names: list[str], replace_ne: bool, aoa: dict | None, multi_process: bool, lower_text: bool):
    global AOA
    AOA = aoa
    for column_name in column_names:
        texts = examples[column_name]
        core_data = generate_core_for_texts(texts, replace_ne=replace_ne, multi_process=multi_process, lower_text=lower_text)
        examples[column_name] = core_data["core"]
    return examples

def replace_column_with_core_data(dataset: Dataset, column_names: list[str], replace_ne: bool, aoa: dict | None, multi_process: bool, lower_text: bool):
    global AOA
    AOA = aoa
    process_fn = partial(_replace_columns_with_core_data, column_names=column_names, replace_ne=replace_ne, aoa=aoa, multi_process=multi_process, lower_text=lower_text)
    dataset = dataset.map(
        process_fn,
        num_proc=4,
        batch_size=BATCH_SIZE,
        batched=True,
        # remove_columns=dataset.column_names,
        writer_batch_size=1000,
        desc="Replacing column with core sentences",
        load_from_cache_file=False,
    )
    print(dataset)
    return dataset