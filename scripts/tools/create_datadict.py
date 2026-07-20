import argparse
from pathlib import Path

from datasets import load_from_disk
from datasets.dataset_dict import DatasetDict


def read_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', dest='input_path', type=str)
    parser.add_argument('--output-path', '-o', dest='output_path', type=str)
    parser.add_argument('--splits', '-s', dest='splits', type=str, nargs='+', help='Splits to process.')
    return parser.parse_args()


def main():
    args = read_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Splits to process: {args.splits}")

    data_dict = {}
    for split in args.splits:
        print(f"Processing split {split}...")
        print(f"Loading dataset from {input_path / split}...")
        data_dict[split] =  load_from_disk(str(input_path / split))
    print(f"Saving processed dataset to {output_path}...") 
    data_dict = DatasetDict(data_dict)
    print(data_dict)
    data_dict.save_to_disk(str(output_path))

if __name__ == "__main__":
    main()