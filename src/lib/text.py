from datasets.arrow_dataset import Dataset

import spacy
from spacy.tokens import Doc


if spacy.prefer_gpu():
    print("Using GPU")
else:
    print("Using CPU")
NLP = spacy.load("en_core_web_trf", disable=["ner", "textcat", "tok2vec"])


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
    doc: Doc = NLP(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def split_texts_to_sentences(texts: list, min_len: int = 0) -> list:
    """
    Splits a list of texts into sentences.

    Args:
        texts (list): A list of input texts to split.
        min_len (int): Minimum length of sentences to keep. Default is 0.

    Returns:
        list: A list of lists, where each inner list contains sentences from the corresponding text.
    """
    docs = NLP.pipe(texts)
    sentences = []
    for doc in docs:
        sentences += [sent.text.strip() for sent in doc.sents if sent.text.strip() if len(sent) > min_len]
    return sentences
