"""
Nonce data handling utilities.
"""
import multiprocessing
from typing import Any
from math import ceil
from functools import partial
import pandas as pd
import random
import string

import spacy
from datasets.arrow_dataset import Dataset

from src.lib.parser import is_content_word
from src.lib.text import safe_texts


CPU_NUM = min(4, multiprocessing.cpu_count())
NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 64
AOA = {}
random.seed(42)

ID_RANGE = 10000


def _random_chars() -> str:
    return ''.join(random.choices(string.ascii_uppercase, k=6))

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


def generate_core_sentence(sent, doc_id: int, unk_id: dict, ent_id: dict, id_candidates: list[int], **config) -> tuple:
    ent_generator= config.get("ent_generator", "")
    unk_generator= config.get("unk_generator", "")
    delimiter = config.get("delimiter", "").strip()
    if len(delimiter) == 2:
        dl, dr = delimiter
    else:
        dl, dr = "", ""

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
        if ent_generator and token.ent_type_:
            # for the processed NE word, get the cid from core_words_id
            # for new NE words, get a new id from id_candidates
            cid = ent_id.get(token_lower, id_candidates.pop())
            id_candidates.insert(0, cid)  # put the used id back to the front of the list
            assert len(id_candidates) > 0, "id_candidates is empty, please check the code."
            if ent_generator == "NONE":
                core_word = token.text
            if ent_generator == "TAG_ID":
                placeholder = token.ent_type_.upper()
                core_word = f"{dl}{placeholder}-{doc_id}-{cid}{dr}"
            elif ent_generator == "ENT_ID":
                placeholder = "ENT"
                core_word = f"{dl}{placeholder}-{doc_id}-{cid}{dr}"
            elif ent_generator == "ENT":
                placeholder = "ENT"
                core_word = f"{dl}{placeholder}{dr}"
            elif ent_generator == "RANDOM":
                core_word = _random_chars()
            else:
                raise ValueError(f"Unknow ne_generator: {ent_generator}")
            core_word += token.whitespace_
            ent_id[token_lower] = cid

            words.append(core_word)
            rp_ne_num += 1
            continue

        # Reject words outside AOA.
        if AOA and token_lower not in AOA and token_lemma not in AOA:
            cid = unk_id.get(token_lower, id_candidates.pop())
            id_candidates.insert(0, cid)  # put the used id back to the front of the list
            if unk_generator == "NONE":
                core_word = token.text
            if unk_generator == "UNK_ID":
                placeholder = "UNK"
                core_word = f"{dl}{placeholder}-{doc_id}-{cid}{dr}"
            elif unk_generator == "UNK_TAG_ID":
                placeholder = "UNK-" + token.tag_.upper()
                core_word = f"{dl}{placeholder}-{doc_id}-{cid}{dr}"
            elif unk_generator == "UNK":
                placeholder = "UNK"
                core_word = f"{dl}{placeholder}{dr}"
            elif ent_generator == "RANDOM":
                core_word = _random_chars()
            else:
                raise ValueError(f"Unknow unk_generator: {ent_generator}")
            core_word += token.whitespace_
            unk_id[token_lower] = cid

            words.append(core_word)
            rp_unk_num += 1
            continue

        # for content words that are not processed
        words.append(token.text_with_ws)

    text = "".join(words)
    return text, content_word_num, rp_ne_num, rp_unk_num


def generate_core_doc(doc, doc_id: int, config: dict) -> tuple:
    rp_ne_num = 0
    rp_unk_num = 0
    content_word_num = 0
    texts = []
    unk_id = {}
    ent_id = {}

    id_candidates = list(range(ID_RANGE))
    random.shuffle(id_candidates)

    token_num = len(doc)
    for sent in doc.sents:
        try:
            t, cn, nn, un= generate_core_sentence(
                sent, doc_id, unk_id, ent_id, id_candidates,
                **config
            )
        except Exception as e:
            print(doc)
            print(texts)
            raise e
        content_word_num += cn
        rp_ne_num += nn
        rp_unk_num += un
        texts.append(t)

    text = "".join(texts)
    return text, token_num, content_word_num, rp_ne_num, rp_unk_num


def generate_core_for_qa(doc_id, question: str, answer: str, aoa: dict, config: dict) -> tuple:
    global AOA
    if aoa and len(aoa) > 0:
        AOA = aoa
    core_q , core_a = question.strip(), answer.strip()
    if core_q:
        doc_q = NLP(question.strip())
        core_q = generate_core_doc(doc_q, doc_id, config)[0]
    if core_a:
        doc_qa = NLP(question.strip() + " " + answer.strip())
        core_qa = generate_core_doc(doc_qa, doc_id, config)[0]
        core_a = " ".join(core_qa.split()[len(core_q.split()):])
    return core_q, core_a


def generate_core_for_texts(texts: list[str], multi_process: bool, lower_text: bool, config: dict) -> dict:
    assert NLP is not None, "NLP should be initialized"
    if lower_text:
        texts = [t.lower() for t in texts]
    if multi_process:
        docs = NLP.pipe(safe_texts(texts, NLP.max_length), batch_size=BATCH_SIZE, n_process=CPU_NUM)
    else:
        docs = NLP.pipe(safe_texts(texts, NLP.max_length), batch_size=BATCH_SIZE)
    ori_texts = []
    core_texts = []
    token_num = []
    content_words_num = []
    replaced_ne_num = []
    replaced_unk_num = []
    for d_id, doc in enumerate(docs):
        core_sentence , tn, cn, nn, un= generate_core_doc(doc, doc_id=d_id, config=config)
        if core_sentence:
            ori_texts.append(doc.text)
            core_texts.append(core_sentence)
            token_num.append(tn)
            content_words_num.append(cn)
            replaced_ne_num.append(nn)
            replaced_unk_num.append(un)
    return {
    "text": ori_texts,
    "core": core_texts,
    "token_num": token_num,
    "content_words_num": content_words_num,
    "replaced_ne_num": replaced_ne_num,
    "replaced_unk_num": replaced_unk_num,
}


def generate_core_for_examples(examples, multi_process: bool, lower_text: bool, config: dict, column_name: str = "text") -> dict:
    texts = examples[column_name]
    # sents = []
    # for text in texts:
    #     sents.extend(split_text_to_sentences(text))
    # sents = examples["text"]
    return generate_core_for_texts(texts, multi_process=multi_process, lower_text=lower_text, config=config)


def generate_core_dataset(
    dataset: Dataset | Any,
    column_name: str,
    aoa: dict | None,
    multi_process: bool,
    lower_text: bool,
    config: dict
):
    global AOA
    AOA = aoa
    batch_number = ceil(dataset.num_rows / BATCH_SIZE)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {BATCH_SIZE}...")

    print("**** Preprocessing...")
    process_fn = partial(
        generate_core_for_examples,
        multi_process=multi_process,
        lower_text=lower_text,
        config=config,
        column_name=column_name,
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


def _replace_columns_with_core_data(examples, column_names: list[str], aoa: dict | None, multi_process: bool, lower_text: bool, config: dict):
    global AOA
    AOA = aoa
    for column_name in column_names:
        texts = examples[column_name]
        core_data = generate_core_for_texts(texts, multi_process=multi_process, lower_text=lower_text, config=config)
        examples[column_name] = core_data["core"]
        examples["token_num"] = core_data["token_num"]
        examples["content_words_num"] = core_data["content_words_num"]
        examples["replaced_ne_num"] = core_data["replaced_ne_num"]
        examples["replaced_unk_num"] = core_data["replaced_unk_num"]
    return examples

def replace_column_with_core_data(dataset: Dataset, column_names: list[str], aoa: dict | None, multi_process: bool, lower_text: bool, config: dict):
    global AOA
    AOA = aoa
    process_fn = partial(_replace_columns_with_core_data, column_names=column_names, aoa=aoa, multi_process=multi_process, lower_text=lower_text, config=config)
    dataset = dataset.map(
        process_fn,
        num_proc=4,
        batch_size=BATCH_SIZE,
        batched=True,
        # remove_columns=dataset.column_names,
        writer_batch_size=1000,
        desc="Generating core sentences",
        load_from_cache_file=False,
    )
    print(dataset)
    return dataset