import sys
import json
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer


TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "left"


def preprocess(example):
    prompt = example["prompt"]
    response = example["answer"]

    prompt_ids = TOKENIZER(prompt, add_special_tokens=False).input_ids
    response_ids = TOKENIZER(response, add_special_tokens=False).input_ids

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    # pad 到 max_length
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


input_path = Path(sys.argv[1])
output_path = sys.argv[2]

train_path = input_path / "train.json"
test_path = input_path / "test.json"

# --- load train ---
print(f"Loading train.json from {train_path}")
train_ds = load_dataset("json", data_files=str(train_path))["train"]

print("Processing train data...")
tokenized_train = train_ds.map(
    preprocess,
    remove_columns=train_ds.column_names,
    desc="Tokenizing train",
)

dataset_dict = {"train": tokenized_train}

# --- load test (optional) ---
if test_path.exists():
    print(f"Loading test.json from {test_path}")
    test_ds = load_dataset("json", data_files=str(test_path))["train"]

    print("Processing test data...")
    tokenized_test = test_ds.map(
        preprocess,
        remove_columns=test_ds.column_names,
        desc="Tokenizing test",
    )

    dataset_dict["test"] = tokenized_test

# --- save ---
print(f"Saving tokenized dataset to {output_path}")
DatasetDict(dataset_dict).save_to_disk(output_path)