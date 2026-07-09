import argparse
from pathlib import Path

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets import concatenate_datasets, load_from_disk

from src.lib.utils import print_args


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets', '-d', dest='datasets', type=str, action='append'
    )
    parser.add_argument(
        '--dataset-limit', '-l', dest='dataset_limit', type=int,
        help='Limit the number of datasets to merge'
    )
    parser.add_argument(
        '--output-path', '-o', dest='out_path', type=str, required=True,
        help='Path to save merged dataset'
    )
    return parser.parse_args()


def main():
    args = read_args()
    print_args(vars(args))
    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    datasets = []
    for data_path in args.datasets:
        print(f"Loading dataset from {data_path}...")
        dataset = load_from_disk(str(data_path))
        print(f"Loaded dataset from {data_path}")
        print(dataset)
        # randomly sample the dataset if dataset_limit is set
        if args.dataset_limit is not None and args.dataset_limit > 0:
            print(f"Limiting dataset to {args.dataset_limit} samples...")
            if isinstance(dataset, Dataset):
                dataset_limit = min(args.dataset_limit, len(dataset))
                dataset = dataset.shuffle(seed=42).select(range(dataset_limit))
            elif isinstance(dataset, DatasetDict):
                for dc in dataset.keys():
                    dataset_limit = min(args.dataset_limit, len(dataset[dc]))
                    dataset[dc] = dataset[dc].shuffle(seed=42).select(range(dataset_limit))
        datasets.append(dataset)

    merged = {}
    if isinstance(datasets[0], Dataset):
        # merge datasets
        sizes = " ".join([str(len(dt)) for dt in datasets])
        print(f"Dataset sizes: {sizes}")
        print("Merging datasets...")
        merged = concatenate_datasets(datasets)
        print(f"Merged Dataset size: {len(merged)}")
    elif isinstance(datasets[0], DatasetDict):
        # merge dataset dicts
        for dc in datasets[0].keys():
            sizes = " ".join([str(len(dt[dc])) for dt in datasets])
            print(f"dataset[{dc}]: {sizes}")
            print(f"Merging datasets {dc}...")
            merged[dc] = concatenate_datasets([dt[dc] for dt in datasets])
            print(f"merged[{dc}]: {len(merged[dc])}")
        merged = DatasetDict(merged)
    else:
        raise TypeError(f"Dataset or DatasetDict is expected. Got {type(datasets[0])}")
    if merged:
        print(f"Saving dataset to {args.out_path}")
        merged = merged.shuffle(seed=42)
        merged.save_to_disk(args.out_path)

if __name__ == "__main__":
    main()
