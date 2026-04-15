from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-path', '-i', dest='input_path', type=str,
    help='Path to json file or dir of json files'
)
parser.add_argument(
    '--output-path', '-o', dest='output_path', type=str
)
parser.add_argument(
    '--tokenizer', '-tk', dest='tokenizer', type=str,
    choices={"gpt2", "Qwen/Qwen3.5-0.8B-Base", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-1.7B"}
)
args = parser.parse_args()

TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer)
TOKENIZER.padding_side = "left"
if TOKENIZER.pad_token_id is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token

def preprocess_mcq(example):
    question = example["prompt"]
    if "options" in example:
        options = example["options"]
    elif "ori_options" in example:
        options = example["ori_options"]
    else:
        raise ValueError("No options field found in example")
    answer = example["answer"]

    texts = [question + " " + opt for opt in options]
    tokenized = [TOKENIZER(t)["input_ids"] for t in texts]

    label = options.index(answer)

    return {
        "input_ids": tokenized,
        "labels": label
    }

print(f"Using tokenizer: {args.tokenizer}")
print(f"EOS ID: {TOKENIZER.eos_token_id}")
print("Loading dataset from:", args.input_path)
dataset = load_dataset("json", data_files=args.input_path)["train"]

print("Dataset loaded. Sample:", dataset[0])
Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
print("Preprocessing dataset...")
dataset = dataset.map(
    preprocess_mcq,
    remove_columns=dataset.column_names,
    num_proc=4  # 多进程加速
)

print(f"Output path: {args.output_path}")
dataset.save_to_disk(args.output_path)