"""
Generate word frequency based on dictionary and corpus.
"""
from datasets import load_dataset
from tqdm import tqdm

import spacy

spacy.cli.download("en_core_web_trf")
nlp = spacy.load("en_core_web_trf")

ds = load_dataset("MAKILINGDING/english_dictionary")["train"]

# take the first 10 data for testing
ds = ds.select(range(10))


def count_token_frequency(text: str) -> dict:
    """
    Count the frequency of each token in the given text based on lemma and part of speech.
    Skip:
        stop words
        punctuation
        numbers

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing the frequency of each token.
    """
    tokens = nlp(text)
    frequency = {}
    for token in tokens:
        if not token.is_stop and not token.is_punct and not token.like_num:
            pos = token.pos_
            frequency[(token.lemma_.lower(), pos)] = frequency.get((token.lemma_.lower(), pos), 0) + 1
    return frequency


frequency_dict = {}
for item in tqdm(ds, total=len(ds)):
    definition = item["definition"]
    _freq = count_token_frequency(definition)
    for token, freq in _freq.items():
        frequency_dict[token] = frequency_dict.get(token, 0) + freq
print(frequency_dict)