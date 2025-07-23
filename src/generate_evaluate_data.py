"""
Generate data for evaluation of number agreement.
Input: a dataset with sentences.
OUtput: A list of dictionaries with sentences and their predictions.
[
    {
        "prompt": "The number of apples is",
        "option1": "three",
        "option2": "four",
        "correct_option": "three"
    },
    ...
]
"""

import argparse
from pathlib import Path
import random

import spacy
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import inflect


from lib.dataset import load_custom_dataset
from lib.text import clean_text, split_texts_to_sentences


spacy.require_gpu()
NLP = spacy.load("en_core_web_trf", disable=["ner", "textcat", "tok2vec"])

INFLECT_ENGINE = inflect.engine()


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', '-dp', dest='data_path', type=str,
        help='Path to the input dataset.'
    )
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str,
        help='Name of the dataset to load.'
    )
    parser.add_argument(
        '--data-type', '-dt', dest='data_type', type=str,
        help='Type of the dataset to load. (e.g., "text", "json")'
    )
    parser.add_argument(
        '--load-from', '-lf', dest='load_from', type=str, default=None,
        help='Load dataset from Hugging Face or local path.'
    )
    parser.add_argument(
        '--limit', '-l', dest='limit', type=int, default=0,
        help='Limit the size of dataset.'
    )
    parser.add_argument(
        '--preprocess', '-p', dest='preprocess', action='store_true',
        help='Whether to preprocess the dataset.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the generated evaluation data.'
    )
    return parser.parse_args()


def map_preprocess_texts(batch: Dataset) -> dict:
    """
    Process a batch of texts to clean and split into sentences.

    Args:
        batch (Dataset): A batch of dataset containing a 'text' column.

    Returns:
        dict: A dictionary with a 'text' key containing the list of cleaned sentences.
    """
    texts = batch['text']
    cleaned_texts = [clean_text(text) for text in texts]
    sentences = split_texts_to_sentences(cleaned_texts)
    return {"text": sentences}


def transform_verb(token):
    # generate an alternative for a given verb token
    # transform lemma to third person singular if the token is in lemma form
    # else return the lemma of the token
    if token.text == token.lemma_:
        converted = token._.inflect("VBZ")
        if converted == token.text or converted is None:
            converted = token._.inflect("VBG")
        return converted or token.text
    else:
        return token.lemma_


def generate_verb_number_agreement_data(batch: Dataset) -> dict:
    """
    Generate data for number agreement evaluation from the dataset.

    Args:
        dataset (Dataset): A dataset containing sentences.
Returns:
        list: A list of dictionaries with prompts and options for number agreement.
    """
    docs = NLP.pipe(batch['text'])
    texts = []
    prompts = []
    option1 = []
    option2 = []
    answers = []
    for doc in docs:
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB" and token.head.i > token.i:
                verb = token.head
                texts.append(doc.text)
                prompts.append(str(doc[:verb.i]))
                answers.append(verb.text)
                op1, op2 = random.sample([verb.text, transform_verb(verb)], 2)
                option1.append(op1)
                option2.append(op2)
                continue

    return {
        "text": texts,
        "prompt": prompts,
        "option1": option1,
        "option2": option2,
        "answer": answers
    }


def main():
    args = read_args()

    # ======== Load dataset =========
    dataset = load_custom_dataset(
        data_path=args.data_path,
        data_name=args.data_name,
        data_type=args.data_type,
        load_from=args.load_from
    )

    if isinstance(dataset, DatasetDict):
        if 'test' in dataset:
            dataset = dataset['test']
        elif 'val' in dataset:
            dataset = dataset['val']
        elif 'validation' in dataset:
            dataset = dataset['validation']
        elif 'train' in dataset:
            dataset = dataset['train']
        else:
            raise ValueError("DatasetDict does not contain 'test', 'train', or 'validation' splits.")

    if args.limit > 0:
        limit = min(args.limit, len(dataset))
        dataset = dataset.select(range(limit))

    # ======== Clean and split texts into sentences =========
    if args.preprocess:
        # Preprocess the dataset if needed
        dataset = dataset.map(
            map_preprocess_texts,
            batched=True,
            batch_size=10,
            remove_columns=dataset.column_names,
            keep_in_memory=False,
            desc="Preprocessing texts"
        )

    # ======== Generate number agreement data =========
    agreement_data = dataset.map(
        generate_verb_number_agreement_data,
        batched=True,
        batch_size=10,
        remove_columns=dataset.column_names,
        keep_in_memory=False,
        desc="Generating number agreement data"
    )

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    agreement_data.save_to_disk(args.out_path)
    agreement_data.select(range(5)).to_json(Path(args.out_path) / "example_number_agreement.json")

    print(f"Generated {len(agreement_data)} samples for number agreement evaluation.")

if __name__ == "__main__":
    main()
