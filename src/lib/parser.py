from spacy.tokens import Token
from typing import Literal


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
