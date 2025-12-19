import sys
import json
import argparse

from pathlib import Path
from datasets import DatasetDict, Dataset
from transformers import GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-path', '-i', dest='input_path', type=str,
    help='Path to json file or dir of json files'
)
parser.add_argument('--output-path', '-output', dest='output_path', type=str,)
parser.add_argument(
    '--tokenizer', '-tk', dest='tokenizer', type=str,
    choices={"gpt2", "Qwen/Qwen3-0.6B-Base", "HuggingFaceTB/SmolLM2-135M"}
)
parser.add_argument(
    '--mask-prompt', '-mp', dest='mask_prompt', action='store_true',
    help='Mask prompt for SFT'
)
args = parser.parse_args()


print(f"Using tokenizer: {args.tokenizer}")
TOKENIZER = GPT2Tokenizer.from_pretrained(args.tokenizer)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "left"

def preprocess(example):
    prompt = example["prompt"]
    response = example["answer"]

    prompt_ids = TOKENIZER(prompt, add_special_tokens=False).input_ids
    response_ids = TOKENIZER(response, add_special_tokens=False).input_ids

    input_ids = prompt_ids + response_ids
    if args.mask_prompt:
        labels = [-100] * len(prompt_ids) + response_ids
    else:
        labels = input_ids

    # pad to max_length
    max_length = 1024
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [TOKENIZER.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
    else:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    attention_mask = [1 if id != TOKENIZER.pad_token_id else 0 for id in input_ids]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


input_path = Path(args.input_path)
output_path = args.output_path
Path(output_path).mkdir(exist_ok=True, parents=True)

train_js = []
if input_path.is_dir():
    for json_file in input_path.rglob("*.json"):
        print(f"Loading train.json from {json_file}")
        with open(json_file, "r") as f:
            train_js += json.load(f)
else:
    print(f"Loading train.json from {input_path}")
    with open(input_path, "r") as f:
        train_js += json.load(f)

# --- load data ---
train_ds = Dataset.from_list(train_js)

print("Processing train data...")
tokenized_train = train_ds.map(
    preprocess,
    remove_columns=train_ds.column_names,
    desc="Tokenizing train",
)

dataset_dict = {"train": tokenized_train}

# --- save ---
print(f"Saving tokenized dataset to {output_path}")
DatasetDict(dataset_dict).save_to_disk(output_path)
