import argparse
from pathlib import Path

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets import concatenate_datasets, load_from_disk

from lib.utils import print_args


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', '-dd', dest='data_dir', type=str, required=True,
        help='Dir of datasets'
    )
    parser.add_argument(
        '--part-range', '-pr', dest='part_range', nargs='+', type=int, required=True,
        help='Range of parts of datasets'
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
    assert len(args.part_range) == 2
    from_part, end_part  = args.part_range
    for part in range(from_part, end_part + 1):
        data_path = Path(args.data_dir) / f"part{part}"
        print(f"Loading dataset from {data_path}...")
        if not data_path.is_dir():
            continue
        try:
            dataset = load_from_disk(str(data_path))
            print(f"Loaded dataset from {data_path}")
            datasets.append(dataset)
        except Exception as e:
            print(f"Failed to load dataset from {data_path}: {e}")
    # for data_path in Path(args.data_dir).iterdir():
    #     if not data_path.is_dir():
    #         continue
    #     try:
    #         dataset = load_from_disk(str(data_path))
    #         print(f"Loaded dataset from {data_path}(size: {len(dataset)})")
    #         datasets.append(dataset)
    #     except Exception as e:
    #         print(f"Failed to load dataset from {data_path}: {e}")

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
        merged.save_to_disk(args.out_path)


if __name__ == "__main__":
    main()
