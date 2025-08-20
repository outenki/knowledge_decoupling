from pathlib import Path
import argparse
from functools import partial

from transformers import GPT2Tokenizer

from lib.dataset import load_custom_dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str,
        help='Dataset path to load from.'
    )
    parser.add_argument(
        '--load-from', '-lf', dest='load_from', type=str, choices={"local", "hf"},
        help='Load data from local or hf(hugging face)'
    )
    parser.add_argument(
        '--data-type', '-dt', dest='data_type', type=str, required=False, default=None,
        help=(
            'Type of the dataset to load.'
            'If not provided, the dataset will be loaded as a Hugging Face Dataset.'
        )
    )
    parser.add_argument(
        '--data-column', '-dc', dest='data_column', type=str, choices=["text", "nonce"],
        help='Column of dataset to use.'
    )
    parser.add_argument(
        '--block-size', '-bs', dest='block_size', type=str, choices=["all", "128", "512", "1024"],
        help='Max length of data'
    )
    parser.add_argument(
        '--output-path', '-o', dest='output_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def tokenize_examples(examples, tokenizer, column_name: str):
    out = tokenizer(examples[column_name], return_attention_mask=False)
    out["labels"] = out["input_ids"]
    return out


def group_texts_to_blocks(examples, block_size: int):
    concatenated = sum(examples["input_ids"], [])
    total_length = len(concatenated)
    total_length = (total_length // block_size) * block_size

    result = {
        "input_ids": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    }
    result["labels"] = result["input_ids"].copy()

    return result


def main():
    args = read_args()
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    datasets = load_custom_dataset(args.data_name, args.data_type, args.load_from)

    # === Tokenize dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    map_func = partial(tokenize_examples, tokenizer=tokenizer, column_name=args.data_column,)

    tokenized_datasets = datasets.map(
        map_func,
        desc="Tokenizing data",
        batched=True,
        remove_columns=[args.data_column]
    )

    # === Slice and concatenate data blocks
    if args.block_size != "all":
        block_size = int(args.block_size)
        map_func = partial(group_texts_to_blocks, block_size=block_size)
        lm_datasets = tokenized_datasets.map(
            map_func, batched=True,
            desc=f"Chunking data to block size {block_size}",
        )
        lm_datasets.save_to_disk(args.output_path)
        return

    for block_size in (128, 512, 1024):
        map_func = partial(group_texts_to_blocks, block_size=block_size)
        lm_datasets = tokenized_datasets.map(
            map_func, batched=True,
            desc=f"Chunking data to block size {block_size}",
        )
        lm_datasets.save_to_disk(args.output_path + f"-bs{block_size}")



if __name__ == "__main__":
    main()
