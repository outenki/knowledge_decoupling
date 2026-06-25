# %%
import argparse
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from pathlib import Path

from src.lib.dataset import load_custom_dataset, select_data_by_indices
from src.lib.dataset import slice_dataset
from src.lib.utils import print_args
from src.data_processing.core_data.lib import generate_core_dataset


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
        '--ala', '-aoa', dest='aoa', type=str, default="", help='Path to aoa data (csv)'
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


def _process_dataset(dataset: Dataset, args):
    dt = slice_dataset(dataset, args.start_from, args.data_limit)
    print(f"Dataset has {dt.num_rows} samples after slicing.")
    out_path = Path(args.out_path)
    processed_dataset = generate_core_dataset(
        dt,
        multi_process=args.multi_process,
        max_n=args.max_n,
        nonce_word_bank=args.nonce_word_bank,
        keep_word_identical=args.keep_word_identical,
    )
    processed_dataset.save_to_disk(str(out_path), max_shard_size="500MB")
     
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


    # ======== Generate nonce sentences ========
    print("**** Processing dataset ...")
    _process_dataset(dataset, args)
    print("Use src/data_processing/merge_dataset.py to merge parts later if you need a single dataset directory.")


if __name__ == "__main__":
    main()
