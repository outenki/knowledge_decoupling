import json
import argparse

from pathlib import Path
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer

from src.lib.dataset import generate_qa_message, format_qa_prompt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-path', '-i', dest='input_path', type=str,
    help='Path to json file or dir of json files'
)
parser.add_argument('--output-path', '-output', dest='output_path', type=str,)
parser.add_argument(
    '--tokenizer', '-tk', dest='tokenizer', type=str,
    choices={"gpt2", "Qwen/Qwen3.5-0.8B-Base", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-1.7B"}
)
parser.add_argument(
    '--apply-chat-template', '-ct', dest='chat_template', action='store_true',
    help='Apply chat template for tokenization'
)
parser.add_argument(
    '--skip-answer', '-sa', dest='skip_answer', action='store_true',
    help='Skip answer field (for unsupervised pretraining or nonce model training).\n Only applies when --apply-chat-template is not set.'
)
parser.add_argument(
    '--mask-prompt', '-mp', dest='mask_prompt', action='store_true',
    help='Mask prompt for SFT'
)
parser.add_argument(
    '--max-length', dest='max_length', type=int, default=1024,
    help='Maximum sequence length after tokenization'
)
args = parser.parse_args()


print(f"Using tokenizer: {args.tokenizer}")
TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer)
TOKENIZER.padding_side = "left"
if TOKENIZER.pad_token_id is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token

# TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})

print(f"EOS ID: {TOKENIZER.eos_token_id}")
print(f"PAD ID: {TOKENIZER.pad_token_id}")
PAD_ID = TOKENIZER.pad_token_id


def truncate_and_pad(input_ids, attention_mask, labels):
    max_length = args.max_length

    if len(input_ids) >= max_length:
        print(f"Data is too long, truncating to max_length: {max_length}")
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [PAD_ID] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return input_ids, attention_mask, labels


def preprocess_chat_template(example):
    messages = generate_qa_message(example)
    # Encoding prompt
    prompt_messages = messages[:-1]
    prompt_out = TOKENIZER.apply_chat_template(
        prompt_messages, 
        tokenize=True, 
        add_generation_prompt=True,
        return_dict=True,  # Some tokenizers return a BatchEncoding-like mapping here.
    )
    prompt_ids = prompt_out["input_ids"]
    prompt_len = len(prompt_ids)

    # Encoding full input (prompt + response)
    full_out = TOKENIZER.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=False,
        return_dict=True,
        truncation=True,
        max_length=args.max_length,
    )
    full_ids = full_out["input_ids"]
    attention_mask = full_out["attention_mask"]

    # Generate labels
    if args.mask_prompt:
        # for SFT, we only calculate loss on the response part, so we set prompt part to -100
        labels = [-100] * len(full_ids)
        labels[prompt_len:] = full_ids[prompt_len:]
    else:
        labels = list(full_ids)

    full_ids, attention_mask, labels = truncate_and_pad(full_ids, attention_mask, labels)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess_concat(example):
    prompt, response = format_qa_prompt(example)

    p_out = TOKENIZER(prompt, add_special_tokens=False)
    r_out = TOKENIZER(response + TOKENIZER.eos_token, add_special_tokens=False)

    prompt_ids = p_out.input_ids if p_out.input_ids is not None else []
    response_ids = r_out.input_ids if r_out.input_ids is not None else []

    if args.skip_answer:
        response_ids = []
    input_ids = prompt_ids + response_ids

    if args.mask_prompt:
        # for SFT, we only calculate loss on the response part, so we set prompt part to -100
        labels = [-100] * len(prompt_ids) + response_ids
    else:
        labels = input_ids

    attention_mask = [1] * len(input_ids)
    input_ids, attention_mask, labels = truncate_and_pad(input_ids, attention_mask, labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess(example):
    if args.chat_template:
        return preprocess_chat_template(example)
    else:
        return preprocess_concat(example)


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
example = train_ds[0]
if args.chat_template:
    print("Chat Template:")
    print(generate_qa_message(example))
else:
    prompt, response = format_qa_prompt(example)
    print("Concat Template:")
    print(prompt + response)

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
