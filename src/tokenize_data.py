from pathlib import Path
import argparse
from functools import partial

from transformers import GPT2Tokenizer

from lib.dataset import load_custom_dataset, tokenize_examples


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', '-dp', dest='data_path', type=str,
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
        '--data-max-length', '-dml', dest='max_length', type=int, help='Max length of data'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def main():
    args = read_args()
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    data = load_custom_dataset(args.data_path, args.data_type, args.load_from)

    # === Tokenize dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenize_func = partial(
        tokenize_examples,
        tokenizer=tokenizer,
        column_name=args.data_column,
        max_length=args.max_length
    )

    tokenized_data = data.map(
        tokenize_func,
        desc="Tokenizing data",
        batched=True,
        batch_size=1000,
        remove_columns=[args.data_column]
    )
    tokenized_data.save_to_disk(args.out_path)


if __name__ == "__main__":
    main()
