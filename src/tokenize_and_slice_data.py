import argparse
from functools import partial
from pathlib import Path
from datasets import Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from lib.dataset import load_custom_dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", "-dn", type=str, required=True)
    parser.add_argument("--load-from", "-lf", type=str, choices=["local", "hf"], required=True)
    parser.add_argument("--data-type", "-dt", type=str, default=None)
    parser.add_argument("--data-column", "-dc", type=str, choices=["text", "nonce"], required=True)
    parser.add_argument("--tokenize", "-t", action="store_true")
    parser.add_argument("--slice", "-s", action="store_true")
    parser.add_argument("--block-size", "-bs", type=int, default=None)
    parser.add_argument("--output-path", "-o", type=str, required=True)
    return parser.parse_args()


def tokenize_examples(examples, tokenizer, column_name: str):
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
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    input_blocks = [concatenated[i:i + block_size] for i in range(0, total_length, block_size)]

    return {"input_ids": input_blocks, "labels": input_blocks}


def main():
    args = read_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(">>> Loading data ...")
    datasets = load_custom_dataset(args.data_name, args.data_type, args.load_from)

    # === TOKENIZATION ===
    tokenized_datasets = datasets
    if args.tokenize:
        print(">>> 开始分词 ...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        map_func = partial(tokenize_examples, tokenizer=tokenizer, column_name=args.data_column)
        tokenized_datasets = datasets.map(
            map_func,
            batched=True,
            num_proc=4,
            remove_columns=[args.data_column],
            desc="Tokenizing data"
        )

        tokenized_path = output_path / "tokenized"
        print(f">>> Save tokenized dataset to: {tokenized_path}")
        tokenized_datasets.save_to_disk(str(tokenized_path))

    # === BINARIZATION / SLICE ===
    if not args.slice:
        print(">>> Skip slicing")
        return

    print(">>> Slicing ...")
    block_sizes = [args.block_size] if args.block_size else [128, 512, 1024]

    for bs in block_sizes:
        print(f"  -> block_size = {bs}")
        map_func = partial(group_texts_to_blocks, block_size=bs)

        lm_datasets = tokenized_datasets.map(
            map_func,
            batched=True,
            batch_size=3000,
            num_proc=4,
            desc=f"Chunking to blocks ({bs})",
            remove_columns=tokenized_datasets["train"].column_names
        )

        bin_path = output_path / f"binarized_bs{bs}"
        print(f"  -> Save to: {bin_path}")
        lm_datasets.save_to_disk(str(bin_path))

    print("✅ Done！")


if __name__ == "__main__":
    main()