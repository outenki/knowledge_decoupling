# %%
import argparse
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from pathlib import Path
import tqdm
import json
import multiprocessing

import tqdm

from src.lib.dataset import load_custom_dataset, select_data_by_indices
from src.lib.dataset import slice_dataset
from src.lib.utils import print_args
from src.lib.nonce_data import generate_nonce_for_dataset

# if spacy.prefer_gpu():
#     print("Using GPU")
# else:
#     print("Using CPU")

CPU_NUM = min(2, multiprocessing.cpu_count())
BATCH_SIZE = 64
NONCE_WORD_BANK = {}


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', dest='data', type=str, help='Dataset path to load from.')
    parser.add_argument('--split', '-sp', dest='split', type=str, default="train", help='Dataset split name to process.') 
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
        '--nonce-word-bank', '-nwb', dest='nonce_word_bank', type=str, default="",
        help='Path to existing nonce word bank.'
    )
    parser.add_argument(
        '--multi-process', '-mp', dest='multi_process', action='store_true',
        help='Use multi-processing for nonce sentence generation.'
    )
    parser.add_argument(
        '--max-matching-number', '-mn', dest='max_n', type=int, default=1,
        help='The max number of matched nonce sentence.'
    )
    parser.add_argument(
        '--keep-word-identical', '-kw', dest='keep_word_identical', action='store_true',
        help='Keep the original word identical when generating nonce sentences.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def _prcocess_dataset(
    dataset: Dataset, start_from: int, data_limit: int, args
) -> Dataset | None:
    dt = slice_dataset(dataset, start_from, data_limit)
    print(f"Dataset has {dt.num_rows} samples after slicing.")
    nonce_word_bank_js = args.nonce_word_bank

    print(f"Loading nonce word bank from {nonce_word_bank_js}...")
    with tqdm.open(nonce_word_bank_js, mode='r', encoding='utf-8', total=None, unit='B', unit_scale=True, desc="加载 JSON") as f:
        nonce_word_bank = json.load(f)

    processed_dataset = generate_nonce_for_dataset(
        dt,
        multi_process=args.multi_process,
        max_n=args.max_n,
        nonce_word_bank=nonce_word_bank,
        keep_word_identical=args.keep_word_identical,
    )
    return processed_dataset


def main():
    args = read_args()
    print_args(vars(args))
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if args.multi_process:
        print("Multi process for NLP.pipe")
    else:
        print("NON-Multi process for NLP.pipe")

    # ========  Load dataset ========
    print("**** Loading dataset...")
    dataset = load_custom_dataset(
        data_name=args.data,
        data_type=None,
        load_from=args.load_from
    )
    print(f"Dataset loaded with {dataset.num_rows} samples.")
    if isinstance(dataset, DatasetDict):
        dataset = dataset[args.split]
        dataset = select_data_by_indices(dataset, args.kept_indices)


    # ======== Generate nonce sentences ========
    print("**** Processing dataset ...")
    dataset = _prcocess_dataset(dataset, args.start_from, args.data_limit, args)
    if dataset:
        print(f"Dataset has {dataset.num_rows} samples after generating nonce sentences.")
        print(f"Saving dataset with nonce sentences to {out_path}...")
        dataset.save_to_disk(out_path)
        dataset.select(range(5)).to_json(Path(args.out_path) / "example_sentences.json")


if __name__ == "__main__":
    main()
