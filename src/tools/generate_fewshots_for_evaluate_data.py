"""
- Load a json data
- Sample data and generate few shot exampls
- Remove the sampled data from the input json data
"""

import json
import random
import argparse
from pathlib import Path


def add_id_to_data(data: list[dict]) -> list[dict]:
    if "id" in data[0]:
        return
    res = []
    for i, d in enumerate(data):
        d["id"] = i
        res.append(d)
    return res


def sample_balanced_examples(data: list[dict], n: int) -> list[dict]:
    answers = set([d["answer"] for d in data])
    res = []
    for _ans in answers:
        _data_ans = [d for d in data if d["answer"] == _ans]
        res += random.sample(_data_ans, n)
    return res


def remove_samples(data: list[dict], samples: list[dict]) -> list[dict]:
    sample_ids = set([d["id"] for d in samples])
    return [d for d in data if d["id"] not in sample_ids]


parser = argparse.ArgumentParser()
parser.add_argument('--input-data', '-i', dest='input_data', type=str, help='Input json data')
parser.add_argument('--output-path', '-o', dest='output_path', type=str, help='Output path')
parser.add_argument('--sample-number', '-n', dest='sample_num', type=int, help='Number of samples')
parser.add_argument('--balanced', '-b', dest='is_balanced', action='store_true', help='Output json data')
args = parser.parse_args()


input_data_fn = Path(args.input_data)
print(f"Loading input data from {input_data_fn}")
with open(input_data_fn, 'r') as f:
    input_data = json.load(f)


output_path = Path(args.output_path)
output_path.mkdir(parents=True, exist_ok=True)

# backup data
ori_data_fn = input_data_fn.stem + ".ori" + input_data_fn.suffix
print(f"Backing up input data to {output_path / input_data_fn}")
with open(output_path / ori_data_fn, "w") as f:
    json.dump(input_data, f, indent=2)


# sample data
print(f"Sampling {args.sample_num} examples...")
input_data = add_id_to_data(input_data)
if args.is_balanced:
    examples = sample_balanced_examples(input_data, args.sample_num)
else:
    examples = random.sample(input_data, args.sample_num)
print(f"{len(examples)} examples sampled.")

data = remove_samples(input_data, examples)

# save results
data_fn = input_data_fn.name
print(f"Saving new input data to {output_path / data_fn}")
with open(output_path / data_fn, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saving example data to {output_path}/examples.json")
with open(output_path / "examples.json", "w") as f:
    json.dump(examples, f, indent=2)
