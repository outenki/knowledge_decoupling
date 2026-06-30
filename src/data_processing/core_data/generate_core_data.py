# %%
import argparse
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from pathlib import Path

import pandas as pd

from src.lib.dataset import load_custom_dataset, select_data_by_indices
from src.lib.dataset import slice_dataset
from src.lib.utils import print_args
from src.data_processing.core_data.lib import generate_core_dataset


def read_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', dest='data', type=str, help='Dataset path to load from.')
    parser.add_argument('--split', '-sp', dest='split', type=str, default="train", help='Dataset split name to process.') 
    parser.add_argument("--kept-indices", "-ki", type=str, default=None, help="Path to json file")
    parser.add_argument('--shuffle', '-sd', dest='shuffle', action='store_true')
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
    parser.add_argument('--aoa', '-aoa', dest='aoa', type=str, default="", help='Path to aoa data (csv)')
    parser.add_argument('--aoa-threshold', '-at', dest='aoa_threshold', type=float, default=0, help='AOA threshold')
    parser.add_argument('--replace-ne', '-rne', dest='replace_ne', action='store_true')
    parser.add_argument(
        '--multi-process', '-mp', dest='multi_process', action='store_true',
        help='Use multi-processing for nonce sentence generation.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def _process_dataset(dataset: Dataset, aoa: dict,args):
    dt = slice_dataset(dataset, args.start_from, args.data_limit)
    print(f"Dataset has {dt.num_rows} samples after slicing.")
    out_path = Path(args.out_path)
    processed_dataset = generate_core_dataset(
        dt, replace_ne=args.replace_ne, aoa=aoa, multi_process=args.multi_process,
    )
    processed_dataset.save_to_disk(str(out_path), max_shard_size="500MB")
    print(f"Dataset has {processed_dataset.num_rows} core sentences.")
     
    processed_dataset.select(range(min(50, len(processed_dataset)))).to_json(
        out_path / "example_sentences.json"
    )


def main():
    args = read_args()
    print_args(vars(args))
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if args.multi_process:
        print("Using spaCy multi-process inside each part.")
    else:
        print("Using single-process spaCy pipeline.")

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
    if args.shuffle:
        dataset.shuffle(seed=42)


    # ========  Load aoa ========
    aoa = {}
    if args.aoa:
        aoa = (
            pd.read_csv(args.aoa, usecols=["Word", "AoA_Kup_lem"])
            .set_index("Word")["AoA_Kup_lem"]
            .to_dict()
        )
        if args.aoa_threshold > 0:
            aoa = {k: v for k, v in aoa.items() if v <= args.aoa_threshold}


    # ======== Generate nonce sentences ========
    print("**** Processing dataset ...")
    _process_dataset(dataset, aoa, args)


if __name__ == "__main__":
    main()
