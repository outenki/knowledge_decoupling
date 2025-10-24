"""
Preprocess dataset script for knowledge decoupling project.
- Clean texts
- Split texts into sentences
- A new DatasetDict is created with train and test splits.
- Each row in the dataset contains one sentence.
"""

import argparse
from pathlib import Path
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

from lib.dataset import load_custom_dataset
from lib.text import clean_text, split_texts_to_sentences


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str,
        help='Dataset path to load from.'
    )
    parser.add_argument(
        '--data-type', '-dt', dest='data_type', type=str, required=False, default=None,
        help=(
            'Type of the dataset to load.'
            'If not provided, the dataset will be loaded as a Hugging Face Dataset.'
        )
    )
    parser.add_argument(
        '--load-from', '-lf', dest='load_from', choices=["hf", "local"],
        help='Load dataset from Hugging Face or local path.'
    )
    parser.add_argument(
        '--limit', '-l', dest='data_limit', type=int, required=False, default=None,
        help='Limit the number of samples to process.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def batch_split_texts_to_sentences(batch: Dataset) -> dict:
    """
    Splits a batch of texts into sentences.

    Args:
        batch (Dataset): A batch of dataset containing a 'text' column.

    Returns:
        dict: A dictionary with a 'text' key containing the list of sentences.
    """
    texts = batch['text']
    sources = batch.get('source', None)
    if sources:
        texts = [t for s, t in zip(sources, texts) if s != "stack_edu" and s != "finemath" and s != "infimm_webmath"]
    cleaned_texts = [clean_text(text) for text in texts]
    sentences = split_texts_to_sentences(cleaned_texts)
    return {"text": sentences}


def main():
    args = read_args()
    print(vars(args))

    # ======== Load dataset =========
    dataset = load_custom_dataset(
        data_name=args.data_name,
        data_type=args.data_type,
        load_from=args.load_from
    )

    # ======== Limit dataset size if specified =========
    if isinstance(dataset, DatasetDict):
        dataset = dataset['train']
    if args.data_limit:
        dataset = dataset.select(range(args.data_limit))

    # ======== Clean and split texts into sentences =========
    processed_dataset = dataset.map(
        batch_split_texts_to_sentences,
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
        keep_in_memory=False,
        desc="Processing texts to sentences"
    )

    _dict = processed_dataset.train_test_split(shuffle=True, test_size=0.01, seed=42)
    processed_dataset_dict = DatasetDict({
        "train": _dict["train"],
        "val": _dict["test"]
    })
    # Save the processed dataset if out_path is provided
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    processed_dataset_dict.save_to_disk(args.out_path)
    print("Processed dataset saved to:", args.out_path)

    # Save some examples to verify
    processed_dataset.select(range(5)).to_json(Path(args.out_path) / "example_sentences.json")
    print(f"Generated {len(processed_dataset)} sentences from the dataset.")


if __name__ == "__main__":
    main()
