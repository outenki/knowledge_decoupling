# %%
import argparse
from collections import defaultdict
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from math import ceil
from pathlib import Path
import json

import tqdm

from src.lib.dataset import load_custom_dataset, load_texts_from_dataset_batch
from src.lib.dataset import slice_dataset, select_data_by_indices
from src.lib.utils import print_args
from src.lib.text import split_text_to_sentences
from src.lib.nonce_data import generate_nonce_word_bank


SKIP_SOURCES = {"finemath", "stack_edu", "infimm_webmath", "stack-edu", "infimm-webmath"}
BANK_BUILD_BATCH_SIZE = 256

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', dest='data', type=str, help='Dataset path to load from.')
    parser.add_argument("--kept-indices", "-ki", type=str, default=None, help="Path to json file")
    parser.add_argument(
        '--load-from', '-lf', dest='load_from', choices=["hf", "local"],
        help='Load dataset from Hugging Face or local path.'
    )
    parser.add_argument(
        '--start-from', '-sf', dest='start_from', type=int, default=0, required=False,
        help='Load data from line.'
    )
    parser.add_argument(
        '--limit', '-l', dest='data_limit', type=int, default=0, required=False,
        help='Limit the number of samples to process. 0 means no limit.'
    )
    parser.add_argument(
        '--multi-process', '-mp', dest='multi_process', action='store_true',
        help='Use multi-processing for nonce sentence generation.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def generate_nonce_bank_from_dataset(dataset: Dataset, multi_process: bool = False) -> dict[str, list[str]]:
    """
    Generates a nonce word bank from a Hugging Face Dataset object.

    Args:
        dataset (Dataset): A Hugging Face Dataset object.
        multi_process (bool): Whether to use multi-processing for nonce sentence generation.

    Returns:
        dict[str, list[str]]: A dictionary mapping nonce words to lists of sentences containing them.
    """
    nonce_bank: dict[str, set] = defaultdict(set)
    total_batches = ceil(len(dataset) / BANK_BUILD_BATCH_SIZE) if len(dataset) else 0

    for batch_idx in tqdm.tqdm(range(total_batches), desc="Generating nonce bank"):
        texts = load_texts_from_dataset_batch(dataset, batch_idx, BANK_BUILD_BATCH_SIZE)
        if not texts:
            continue

        sentences = []
        for text in texts:
            sentences.extend(split_text_to_sentences(text))
        if not sentences:
            continue

        batch_bank = generate_nonce_word_bank(sentences, multi_process)
        for morph, words in batch_bank.items():
            nonce_bank[morph].update(words)

    return {morph: list(words) for morph, words in nonce_bank.items()}


def main():
    args = read_args()
    print_args(vars(args))

    print("**** Loading dataset...")
    dataset = load_custom_dataset(
        data_name=args.data,
        data_type=None,
        load_from=args.load_from
    )
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    print(f"Dataset loaded with {dataset.num_rows} samples.")

    # filter out sources that are not suitable for nonce sentence generation
    dataset = select_data_by_indices(dataset, args.kept_indices)
    # dataset = skip_dataset_by_column(dataset, "source", SKIP_SOURCES)
    # print(f"Dataset filtered to {dataset.num_rows} samples after skipping sources: {SKIP_SOURCES}.")

    dt = slice_dataset(dataset, args.start_from, args.data_limit)
    nonce_bank = generate_nonce_bank_from_dataset(dt, args.multi_process)
    
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(nonce_bank, f, indent=4)
    print(f"Nonce bank saved to {out_path}.")


if __name__ == "__main__":
    main()