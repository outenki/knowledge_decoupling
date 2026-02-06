import argparse
from functools import partial
from pathlib import Path
from itertools import chain

from transformers import AutoTokenizer

from lib.dataset import load_custom_dataset, slice_dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=False, default="gpt2")
    parser.add_argument("--data-name", "-dn", type=str, required=True)
    parser.add_argument("--load-from", "-lf", type=str, choices=["local", "hf"], required=True)
    parser.add_argument("--data-type", "-dt", type=str, default=None)
    parser.add_argument("--data-column", "-dc", type=str, choices=["text", "nonce"], required=True)
    parser.add_argument("--data-split", "-ds", type=str, required=True, help="train/dev/test")
    parser.add_argument("--tokenize", "-t", action="store_true")
    parser.add_argument("--slice", "-s", action="store_true")
    parser.add_argument("--block-size", "-bs", type=int, required=True)
    parser.add_argument(
        '--start-from', '-sf', dest='start_from', type=int, default=0, required=False,
        help='Load data from line.'
    )
    parser.add_argument(
        '--limit', '-l', dest='data_limit', type=int, default=0, required=False,
        help='Limit the number of samples to process. 0 means no limit.'
    )
    parser.add_argument("--output-path", "-o", type=str, required=True)
    return parser.parse_args()


def tokenize_examples(examples, tokenizer, column_name: str, padding: bool, max_length: int):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
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
        )
        result["labels"] = [ids[:] for ids in result["input_ids"]]
    return result


def group_texts_to_blocks(examples, block_size: int):
    # 拼接为一个长序列后再切块
    concatenated = list(chain.from_iterable(examples["input_ids"]))
    total_length = (len(concatenated) // block_size) * block_size
    input_blocks = [concatenated[i:i + block_size] for i in range(0, total_length, block_size)]

    return {"input_ids": input_blocks, "labels": input_blocks}


def main():
    args = read_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(">>> Loading data ...")
    datasets = load_custom_dataset(args.data_name, args.data_type, args.load_from)
    if isinstance(datasets, dict):
        dataset = datasets[args.data_split]
    else:
        dataset = datasets
    print(f"  -> Dataset size: {len(dataset)}")

    dataset = dataset.shuffle(seed=42)
    dataset = slice_dataset(dataset, args.start_from, args.data_limit)

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
            num_proc=4,
            remove_columns=[args.data_column],
            desc="Tokenizing data"
        )
        if not args.slice:
            tokenized_path = output_path
            print(f">>> Save tokenized dataset to: {tokenized_path}")
            tokenized_dataset.save_to_disk(str(tokenized_path))

    # === BINARIZATION / SLICE ===
    if args.slice:

        print(">>> Slicing ...")
        block_sizes = [args.block_size] if args.block_size else [128, 512, 1024]

        for bs in block_sizes:
            print(f"  -> block_size = {bs}")
            map_func = partial(group_texts_to_blocks, block_size=bs)

            lm_dataset = tokenized_dataset.map(
                map_func,
                batched=True,
                batch_size=3000,
                num_proc=4,
                desc=f"Chunking to blocks ({bs})",
                remove_columns=tokenized_dataset.column_names
            )

            bin_path = output_path / f"bs_{bs}"
            print(f"  -> Save to: {bin_path}")
            lm_dataset.save_to_disk(str(bin_path))

    print("✅ Done！")


if __name__ == "__main__":
    main()
