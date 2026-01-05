from spacy.lang.en import English
from spacy.tokens import Doc
import re

import nltk
from nltk.tokenize import sent_tokenize


NLP = English()
NLP.add_pipe("sentencizer")


def clean_text(text: str) -> str:
    """
    Cleans the input text by removing leading and trailing whitespace,
    and replacing multiple spaces with a single space.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    return ' '.join(text.split())


def split_text_to_sentences(text: str) -> list:
    """
    Splits the input text into sentences using spaCy.

    Args:
        text (str): The input text to split.

    Returns:
        list: A list of sentences.
    """
    # doc: Doc = NLP(text)
    # return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    texts = text.split("\n")

    sents = []
    for _t in texts:
        if _t.strip():
            sents += sent_tokenize(_t)
    return sents


def split_texts_to_sentences(texts: list, min_len: int = 0) -> list:
    """
    Splits a list of texts into sentences.

    Args:
        texts (list): A list of input texts to split.
        min_len (int): Minimum length of sentences to keep. Default is 0.

    Returns:
        list: A list of lists, where each inner list contains sentences from the corresponding text.
    """
    all_sentences = []

    for text in texts:
        if len(text) <= NLP.max_length:
            doc = NLP(text)
            all_sentences.extend([sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > min_len])
        else:
            # 超长文本，按块处理
            chunk_size = NLP.max_length - 1000  # 留出余量
            for start in range(0, len(text), chunk_size):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]
                doc = NLP(chunk)
                all_sentences.extend([sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > min_len])
    return all_sentences


def simple_split_text(text: str) -> list[str]:
    sents = re.split(r'(?<=[。！？.!?\n])\s*', text)
    return [s for s in sents if s]


def simple_split_texts(texts: list) -> list[str]:
    res = []
    for text in texts:
        res.extend(simple_split_text(text))
    return res
