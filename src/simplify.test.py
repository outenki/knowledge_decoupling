# simplify_preserve_tense.py
import spacy
import nltk
import argparse
from pathlib import Path
from functools import partial
import os
import ssl

from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from nltk.corpus import wordnet as wn
from lemminflect import getInflection

from lib.basic_words.basic_words_850 import BASIC_WORDS_850
from lib.basic_words.oxford_3000 import OXFORD_3000
from lib.dataset import load_custom_dataset, load_texts_from_dataset_batch, slice_dataset


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Python < 2.7.9 没有 ssl._create_unverified_context
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
nltk.download('omw-1.4')


nlp = spacy.load("en_core_web_sm")


def spacy_to_wordnet_pos(spacy_pos):
    if spacy_pos.startswith("N"):
        return wn.NOUN
    elif spacy_pos.startswith("V"):
        return wn.VERB
    elif spacy_pos.startswith("J"):
        return wn.ADJ
    elif spacy_pos.startswith("R"):
        return wn.ADV
    else:
        return None


def get_simple_candidate(token, basic_vocab):
    if token.is_punct or token.is_space:
        return token.text

    # skip numbers
    if token.pos_ == "PROPN" or token.like_num:
        return token.text

    # skip NEs
    if token.ent_type_:
        return token.text

    lemma = token.lemma_.lower()
    if lemma in basic_vocab:
        return lemma

    candidates = set()
    for syn in wn.synsets(lemma):
        if not syn:
            continue
        spacy_pos = spacy_to_wordnet_pos(token.pos_)
        if spacy_pos and syn.pos() and spacy_pos != syn.pos():
            continue
        for sn in syn.lemma_names():
            if "_" in sn:
                continue
            if str(token) == "game" or str(token) == "wrote":
                print(f"cand of {token}:", sn)
            if sn == "back" or sn == "pen":
                import ipdb; ipdb.set_trace()
            if sn in basic_vocab:
                candidates.add(sn)

    if not candidates:
        return None

    # take the shortest one
    best = sorted(candidates, key=lambda x: (len(x.split()), len(x)))[0]
    return best


def inflect_candidate(lemma, target_tag):
    if lemma is None:
        return None
    valid_tags = set([
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS",
        "NNP", "NNPS", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"
    ])
    if target_tag not in valid_tags:
        return lemma
    infl = getInflection(lemma, tag=target_tag)
    if infl:
        return infl[0]
    return lemma


def simplify_sentence(sentence, basic_vocab):
    doc = nlp(sentence)
    out_tokens = []
    for token in doc:
        if str(token).lower() in ("be", "am", "are", "is", "were", "was", "been"):
            out_tokens.append(token.text_with_ws)
            continue

        candidate = get_simple_candidate(token, basic_vocab)
        if candidate is None:
            return None
            # out_tokens.append(token.text_with_ws)
            # continue

        inflected = inflect_candidate(candidate, token.tag_)
        if inflected is None:
            inflected = candidate

        if token.text[0].isupper():
            inflected = inflected.capitalize()

        out_tokens.append(inflected + token.whitespace_)
    return "".join(out_tokens)


if __name__ == "__main__":
    examples = [
        'The game was directed by Katsumi Yokota and produced by Tetsuya Mizuguchi.',
        'He wrote and directed Bad Company (1931) with Twelvetrees, and Prestige (1931), and just directed Panama Flo (1932) with Twelvetrees.'
    ]
    for s in examples:
        print("原句: ", s)
        print("简化: ", simplify_sentence(s, OXFORD_3000))
        print("---")