"""
Read JSON files, extract specific columns and generate dataset
"""
import argparse
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json-file', '-jf', dest='json_files', type=str, action='append',
        help='JSON files.'
    )
    parser.add_argument(
        '--column-name', '-cn', dest='column_name', type=str, default="text"
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset.'
    )
    return parser.parse_args()



args = read_args()

# load
column_data = []
for fn in args.json_files:
    print(f"Loading from {fn}")
    with open(fn, "r") as f:
        js = json.load(f)
    for d in tqdm(js, desc=f"Extracting {args.column_name}"):
        column_data.append(d[args.column_name])

data_dict = {args.column_name: column_data}
dataset = Dataset.from_dict(data_dict)


print("---")
print(f"Data size: {len(dataset)}")
print(f"Data example: {dataset[0]}")

dataset_dict = DatasetDict({"train": dataset})

dataset_dict.save_to_disk(args.out_path)
