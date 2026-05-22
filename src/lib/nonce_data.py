"""
Nonce data handling utilities.
"""
import multiprocessing
import itertools
from typing import Any
from math import ceil
from functools import partial

import spacy
from spacy.tokens import Token
import random
from datasets.arrow_dataset import Dataset

from src.lib.parser import extract_token_morph_features, is_content_word, is_vowel
from src.lib.text import safe_texts


CPU_NUM = min(4, multiprocessing.cpu_count())
NLP = spacy.load("en_core_web_sm")
BATCH_SIZE = 64
NONCE_WORD_BANK = {}
random.seed(42)


def serialize_morph(morph_tuple) -> str:
    # morph_tuple: (pos, tag, dep, ent, dir, morph)
    return "|".join([str(i) for i in morph_tuple])


def merge_nonce_banks(bank1: dict, bank2: dict) -> dict:
    """
    Merges two nonce word banks by concatenating the lists of words for each morph feature.
    """
    merged_bank = bank1.copy()
    for morph, words in bank2.items():
        if morph in merged_bank:
            merged_bank[morph].extend(words)
            # remove duplicates
            merged_bank[morph] = list(set(merged_bank[morph]))
        else:
            merged_bank[morph] = words
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
        docs = NLP.pipe(texts, batch_size=64, n_process=CPU_NUM)
    else:
        docs = NLP.pipe(texts, batch_size=64)

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


def match_nonce_words(token: Token, max_n: int, keep_word_identical: bool) -> list[str]:
    """
    Matches a token with a list of words based on its pos and morph features

    Args:
        token (Token): A spaCy Token object.
        max_n (int): The maximum number of nonce sentences to return.
        keep_word_identical (bool): A word is always replaced by the same nonce word.
    Returns:
        list: A list of matched words.
    """
    text, lemma, morph = extract_token_morph_features(token)
    key = serialize_morph(morph)
    candidates = NONCE_WORD_BANK.get(key, [])
    # shuffle candidates
    if not keep_word_identical:
        random.shuffle(candidates)
    if not candidates:
        return []
    try:
        start_index = candidates.index((text, lemma)) + 1
    except ValueError:
        return []

    nonce_words = []
    # for nonce_text, nonce_lemma in candidates:
    for i in range(len(candidates)):
        # 
        nonce_text, nonce_lemma = candidates[(start_index + i) % len(candidates)]
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


def generate_nonce_sentence(doc, max_n: int, keep_word_identical: bool) -> list:
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
    matched_content_word_num = 0
    for token in content_words:
        candidates = match_nonce_words(token, max_n, keep_word_identical)
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
    nonce_words = random.sample(list(itertools.product(*nonce_words_per_token)), k=max_n)
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


def generate_nonce_for_examples(examples, multi_process: bool, max_n: int, keep_word_identical: bool):
    assert NLP is not None, "NLP should be initialized"

    nonce = []
    if multi_process:
        docs = NLP.pipe(safe_texts(examples["text"], NLP.max_length), batch_size=64, n_process=CPU_NUM)
    else:
        docs = NLP.pipe(safe_texts(examples["text"], NLP.max_length), batch_size=64)
    nonce = []
    for doc in docs:
        nonce.append(generate_nonce_sentence(doc, max_n, keep_word_identical))
    return nonce


def generate_nonce_for_dataset(
    dataset: Dataset | Any,
    multi_process: bool,
    max_n: int,
    keep_word_identical: bool,
    nonce_word_bank: dict
):
    batch_number = ceil(dataset.num_rows / BATCH_SIZE)
    print(f"***Processing {dataset.num_rows} samples in {batch_number} batches of size {BATCH_SIZE}...")

    global NONCE_WORD_BANK
    NONCE_WORD_BANK = nonce_word_bank
    print("**** Preprocessing...")
    print("**** Generating nonce sentence...")
    process_fn = partial(generate_nonce_for_examples, multi_process=multi_process, max_n=max_n, keep_word_identical)
    dataset = dataset.map(
        process_fn,
        num_proc=CPU_NUM,
        batch_size=BATCH_SIZE,
        batched=True,
        writer_batch_size=1000,
        desc="Generating nonce sentences"
    )
    dataset = dataset.filter(lambda x: x["nonce"] != "")
    dataset = dataset.shuffle(seed=42)

    print(f"Generated {len(dataset)} samples with nonce sentences.")
    return dataset