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

from lib.dataset import load_custom_dataset, skip_dataset_by_column
from lib.text import clean_text, split_texts_to_sentences
from lib.utils import print_args


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
        '--skip-key', '-sk', dest='skip_key',
        help='Skip data based on column'
    )
    parser.add_argument(
        '--skip-values', '-sv', nargs='+', dest='skip_values',
        help='Skip data based on values of the skip_key'
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
    cleaned_texts = [clean_text(text) for text in texts]
    sentences = split_texts_to_sentences(cleaned_texts)
    return {"text": sentences}


def process_dataset(dataset: Dataset, args):
    if args.data_limit:
        dataset = dataset.select(range(args.data_limit))

    if args.skip_key and args.skip_values:
        dataset = skip_dataset_by_column(dataset, args.skip_key, args.skip_values)

    # ======== Clean and split texts into sentences =========
    return dataset.map(
        batch_split_texts_to_sentences,
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
        keep_in_memory=False,
        desc="Processing texts to sentences"
    )


def main():
    args = read_args()
    print_args(vars(args))

    # ======== Load dataset =========
    dataset = load_custom_dataset(
        data_name=args.data_name,
        data_type=args.data_type,
        load_from=args.load_from
    )

    # ======== Limit dataset size if specified =========
    if isinstance(dataset, DatasetDict):
        processed_dataset_dict = DatasetDict({
            k: process_dataset(d, args) for k, d in dataset.items()
        })
    else:
        processed_dataset_dict = DatasetDict({
            "train": process_dataset(dataset, args)
        })
    

    # Save the processed dataset if out_path is provided
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    processed_dataset_dict.save_to_disk(args.out_path)
    print("Processed dataset saved to:", args.out_path)

    # Save some examples to verify
    for k in processed_dataset_dict:
        processed_dataset_dict[k].select(range(5)).to_json(Path(args.out_path) / "example_sentences.json")
        return


if __name__ == "__main__":
    main()
