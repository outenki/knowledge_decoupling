# %%
import argparse
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from pathlib import Path

import pandas as pd

from src.lib.dataset import load_custom_dataset, select_data_by_indices, maybe_shuffle_dataset
from src.lib.dataset import slice_dataset
from src.lib.utils import print_args
from src.data_processing.core_data.lib import generate_core_dataset, replace_column_with_core_data, load_aoa


def read_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', dest='dataset', type=str, help='Dataset path to load from.')
    parser.add_argument('--columns', '-c', dest='columns', type=str, nargs='+', help='Column names to process.')
    parser.add_argument('--split', '-sp', dest='split', type=str, default="train", help='Dataset split name to process.') 
    parser.add_argument("--kept-indices", "-ki", type=str, default=None, help="Path to json file")
    parser.add_argument('--shuffle', '-sd', dest='shuffle', action='store_true')
    parser.add_argument('--lower-text', '-lower', dest='lower_text', action='store_true')
    parser.add_argument('--inline-replace', '-ir', dest='inline_replace', action='store_true')
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
    parser.add_argument('--ent-generator', dest='ent_generator', choices={"ENT", "NE", "RANDOM"}) 
    parser.add_argument('--unk-generator', dest='unk_generator', choices={"UNK", "UNK-TAG", "RANDOM"}) 
    parser.add_argument('--core-delimiter', dest='core_delimiter') 
    parser.add_argument(
        '--multi-process', '-mp', dest='multi_process', action='store_true',
        help='Use multi-processing for nonce sentence generation.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def _process_dataset(dataset: Dataset, column_names: list[str], aoa: dict, args ):
    print(f"Dataset has {dataset.num_rows} samples before slicing.")
    dt = slice_dataset(dataset, args.start_from, args.data_limit)
    print(f"Dataset has {dt.num_rows} samples after slicing.")
    out_path = Path(args.out_path)
    if not column_names or len(column_names) == 0:
        column_names = ["text"]
    config = {
        "delimiter": args.core_delimiter,
        "ent_generator": args.ent_generator,
        "unk_generator": args.unk_generator,
    }
    if args.inline_replace:
        processed_dataset = replace_column_with_core_data(
            dt, column_names=column_names, aoa=aoa, multi_process=args.multi_process, lower_text=args.lower_text, config=config
        )
    else:
        processed_dataset = generate_core_dataset(
            dt, aoa=aoa, multi_process=args.multi_process, column_name=column_names[0], lower_text=args.lower_text, config=config
        )
    print(f"Dataset has {processed_dataset.num_rows} core sentences.")
     
    processed_dataset.select(range(min(50, len(processed_dataset)))).to_json(
        out_path / "example_sentences.json"
    )
    processed_dataset.save_to_disk(str(out_path), max_shard_size="500MB")


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
    dataset: Dataset | DatasetDict = load_custom_dataset(
        data_name=args.dataset,
        data_type=None,
        load_from=args.load_from
    )
    print(f"Dataset loaded:")
    print(dataset)


    if isinstance(dataset, DatasetDict):
        dataset = dataset[args.split]

    if args.kept_indices is not None:
        print(f"Selecting kept indices from split {args.split}...")
        dataset = select_data_by_indices(dataset, args.kept_indices)

    dataset = maybe_shuffle_dataset(dataset, shuffle=args.shuffle, seed=42)

    # ========  Load aoa ========
    aoa = {}
    if args.aoa:
        aoa = load_aoa(args.aoa, args.aoa_threshold)


    # ======== Generate nonce sentences ========
    print("**** Processing dataset ...")
    assert isinstance(dataset, Dataset), "Dataset must be a Dataset object."
    _process_dataset(dataset, aoa=aoa, args=args, column_names=args.columns)


if __name__ == "__main__":
    main()
