import argparse
from pathlib import Path

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets import concatenate_datasets

from lib.dataset import load_custom_dataset, slice_dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', '-dd', dest='data_dir', type=str, required=True,
        help='Dir of datasets'
    )
    parser.add_argument(
        '--dataset-names', '-dn', dest='dataset_names', type=str, required=True,
        help='Names of datasets to merge. Split names with `,` (no spaces)'
    )
    parser.add_argument(
        '--output-path', '-o', dest='out_path', type=str, required=True,
        help='Path to save merged dataset'
    )
    return parser.parse_args()


def main():
    args = read_args()
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    dataset_names = args.dataset_names.split(",")

    datasets = []
    for dn in dataset_names:
        data_path = Path(args.data_dir) / dn
        dataset = load_custom_dataset(data_path, None, "local")
        print(f"Loading dataset from {data_path}(size: {len(dataset)})")
        datasets.append(dataset)

    merged = {}
    if isinstance(datasets[0], Dataset):
        # merge datasets
        sizes = " ".join([str(len(dt)) for dt in datasets])
        print(f"Dataset sizes: {sizes}")
        merged = concatenate_datasets(datasets)
        print(f"Merged Dataset size: {len(merged)}")
    if isinstance(datasets[0], DatasetDict):
        # merge dataset dicts
        for dc in datasets[0].keys():
            sizes = " ".join([str(len(dt[dc])) for dt in datasets])
            print(f"dataset[{dc}]: {sizes}")
            merged[dc] = concatenate_datasets([dt[dc] for dt in datasets])
            print(f"merged[{dc}]: {len(merged[dc])}")
        merged = DatasetDict(merged)
    else:
        raise TypeError(f"Dataset or DatasetDict is expected. Got {type(datasets[0])}")
    if merged:
        merged.save_to_disk(args.out_path)


if __name__ == "__main__":
    main()
