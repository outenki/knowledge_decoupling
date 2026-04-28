import argparse
from itertools import chain
from functools import partial
from pathlib import Path
import json

from datasets import Dataset
from transformers import AutoTokenizer

from src.lib.dataset import load_custom_dataset, slice_dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=False, default="gpt2")
    parser.add_argument("--data-name", "-dn", type=str, required=True)
    parser.add_argument("--load-from", "-lf", type=str, choices=["local", "hf"], required=True)
    parser.add_argument("--data-type", "-dt", type=str, default=None)
    parser.add_argument("--data-column", "-dc", type=str, choices=["text", "nonce"], default="text")
    parser.add_argument("--data-split", "-ds", type=str, required=True, help="train/dev/test")
    parser.add_argument("--tokenize", "-t", action="store_true")
    parser.add_argument("--slice", "-s", action="store_true")
    parser.add_argument("--block-size", "-bs", type=int, required=True)
    parser.add_argument(
        "--kept-indices", "-ki", type=str, default=None,
        help="Path to json file"
    )
    parser.add_argument(
        '--start-from', '-sf', dest='start_from', type=int, default=0, required=False,
        help='Start offset before shuffling.'
    )
    parser.add_argument(
        '--limit', '-l', dest='data_limit', type=int, default=0, required=False,
        help='Limit the number of samples to process. 0 means no limit.'
    )
    parser.add_argument(
        "--num-proc", "-np", type=int, default=4,
        help="Number of processes used by dataset.map."
    )
    parser.add_argument(
        "--tokenize-batch-size", "-tbs", type=int, default=2048,
        help="Batch size for tokenization."
    )
    parser.add_argument(
        "--slice-batch-size", "-sbs", type=int, default=2048,
        help="Batch size for grouping tokens into fixed-size blocks."
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42,
        help="Shuffle seed applied after slicing."
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Disable shuffling after slicing."
    )
    parser.add_argument("--output-path", "-o", type=str, required=True)
    return parser.parse_args()


def validate_args(args):
    if not args.tokenize and not args.slice:
        raise ValueError("At least one of --tokenize or --slice must be specified.")
    if args.block_size <= 0:
        raise ValueError("--block-size must be a positive integer.")
    if args.num_proc <= 0:
        raise ValueError("--num-proc must be a positive integer.")
    if args.tokenize_batch_size <= 0:
        raise ValueError("--tokenize-batch-size must be a positive integer.")
    if args.slice_batch_size <= 0:
        raise ValueError("--slice-batch-size must be a positive integer.")


def ensure_tokenizer_padding(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def tokenize_examples(examples, tokenizer, column_name: str, padding: bool, max_length: int):
    ensure_tokenizer_padding(tokenizer)
    if padding:
        result = tokenizer(
            examples[column_name],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )
        result["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in ids]
            for ids in result["input_ids"]
        ]
    else:
        result = tokenizer(
            examples[column_name],
            padding=False,
            truncation=False,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        if tokenizer.eos_token_id is not None:
            result["input_ids"] = [
                ids + [tokenizer.eos_token_id]
                for ids in result["input_ids"]
            ]
        result["labels"] = [ids[:] for ids in result["input_ids"]]
    return result


def group_texts_to_blocks(examples, block_size: int):
    # 拼接为一个长序列后再切块
    concatenated = list(chain.from_iterable(examples["input_ids"]))
    total_length = (len(concatenated) // block_size) * block_size
    input_blocks = [concatenated[i:i + block_size] for i in range(0, total_length, block_size)]
    attention_blocks = [[1] * block_size for _ in input_blocks]

    return {
        "input_ids": input_blocks,
        "attention_mask": attention_blocks,
        "labels": [ids[:] for ids in input_blocks],
    }


def main():
    args = read_args()
    validate_args(args)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(">>> Loading data ...")
    datasets = load_custom_dataset(args.data_name, args.data_type, args.load_from)
    if isinstance(datasets, dict):
        dataset = datasets[args.data_split]
    else:
        dataset = datasets
    print(f"  -> Dataset size: {len(dataset)}")

    kept_indices = None
    if not Path(args.kept_indices).is_file():
        print(f"Warning: kept indices file not found at {args.kept_indices}. All examples will be kept.")
    else:
        with open(args.kept_indices, "r") as f:
            kept_indices = json.load(f)
        print(f"  -> {len(kept_indices)} examples will be kept based on the provided indices.")
        print(">>> Filtering dataset based on kept indices ...")
        dataset = dataset.select(kept_indices)
        print(f"  -> Dataset size after filtering: {len(dataset)}")

    dataset = slice_dataset(dataset, args.start_from, args.data_limit)
    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=args.shuffle_seed)

    # === TOKENIZATION ===
    tokenized_dataset = dataset
    if args.tokenize:
        print(">>> Tokenizing ...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        padding = True
        if args.slice:
            padding = False

        map_func = partial(
            tokenize_examples,
            tokenizer=tokenizer,
            column_name=args.data_column,
            padding=padding,
            max_length=args.block_size
        )
        tokenized_dataset = dataset.map(
            map_func,
            batched=True,
            batch_size=args.tokenize_batch_size,
            num_proc=args.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing data"
        )
        print(f"  -> {len(tokenized_dataset)} samples after tokenization.")
        tokenizer.save_pretrained(output_path / "tokenizer")
        print(f"  -> Tokenizer saved to: {output_path / 'tokenizer'}")
        if not args.slice:
            tokenized_path = output_path
            print(f">>> Save tokenized dataset to: {tokenized_path}")
            tokenized_dataset.save_to_disk(str(tokenized_path))

    # === BINARIZATION / SLICE ===
    if args.slice:
        print(f">>> Concact and Slice to blocks with size {args.block_size}...")
        print(f"  -> block_size = {args.block_size}")
        map_func = partial(group_texts_to_blocks, block_size=args.block_size)

        lm_dataset = tokenized_dataset.map(
            map_func,
            batched=True,
            batch_size=args.slice_batch_size,
            num_proc=args.num_proc,
            desc=f"Chunking to blocks ({args.block_size})",
            remove_columns=tokenized_dataset.column_names
        )

        bin_path = output_path
        print(f"  -> Save to: {bin_path}")
        lm_dataset.save_to_disk(str(bin_path))

    print("✅ Done！")


if __name__ == "__main__":
    main()
