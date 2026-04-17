#!/usr/bin/env python3
import argparse
from pathlib import Path
import struct

from datasets import load_from_disk, Dataset, DatasetDict

def export_binary(dataset: Dataset, output_dir: Path, split_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{split_name}.bin"
    idx_path = output_dir / f"{split_name}.idx"

    print(f"⚙️ Exporting binary dataset to {bin_path}")

    offsets = []
    offset = 0
    with open(bin_path, "wb") as f_bin:
        for tokens in dataset["input_ids"]:
            f_bin.write(struct.pack(f"<{len(tokens)}I", *tokens))
            offset += len(tokens) * 4  # 每个 int32 占 4 字节
            offsets.append(offset)

    with open(idx_path, "wb") as f_idx:
        f_idx.write(struct.pack(f"<{len(offsets)}Q", *offsets))

    print(f"✅ Saved {len(offsets)} sequences → {bin_path.name}, {idx_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", "-d", type=str, required=True,
        help="Path to tokenized dataset (Dataset or DatasetDict)"
    )
    parser.add_argument(
        "--output-path", "-o", type=str, required=True,
        help="Output directory for .bin/.idx files"
    )
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    output_dir = Path(args.output_path)

    if isinstance(dataset, DatasetDict):
        for split_name, split_dataset in dataset.items():
            export_binary(split_dataset, output_dir, split_name)
    elif isinstance(dataset, Dataset):
        export_binary(dataset, output_dir, "train")
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")


if __name__ == "__main__":
    main()