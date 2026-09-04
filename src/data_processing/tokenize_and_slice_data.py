import argparse
from itertools import chain
from functools import partial
from pathlib import Path
import json
import re
import string

from transformers import AutoTokenizer

from src.lib.dataset import load_custom_dataset, slice_dataset, maybe_shuffle_dataset
from src.data_processing.core_data.lib import load_aoa


AOA_THRESHOLD = 10
CONTRACTIONS = {
    # be
    "I'm",
    "you're",
    "he's",
    "she's",
    "it's",
    "we're",
    "they're",

    # have
    "I've",
    "you've",
    "he's",
    "she's",
    "it's",
    "we've",
    "they've",

    # will
    "I'll",
    "you'll",
    "he'll",
    "she'll",
    "it'll",
    "we'll",
    "they'll",

    # would
    "I'd",
    "you'd",
    "he'd",
    "she'd",
    "it'd",
    "we'd",
    "they'd",

    # negative
    "isn't",
    "aren't",
    "wasn't",
    "weren't",

    "haven't",
    "hasn't",
    "hadn't",

    "don't",
    "doesn't",
    "didn't",

    "can't",
    "couldn't",
    "won't",
    "wouldn't",
    "shan't",
    "shouldn't",
    "mustn't",
    "mightn't",
    "needn't",
    "daren't",
    "oughtn't",

    # let us
    "let's",

    # common pronoun / question-word contractions
    "that's",
    "that's",
    "what's",
    "who's",
    "where's",
    "when's",
    "why's",
    "how's",

    "there's",
    "here's",

    # other common contractions
    "what're",
    "who're",
    "where're",
    "when're",
    "how're",

    "what've",
    "who've",
    "where've",
    "how've",

    "what'll",
    "who'll",
    "where'll",
    "when'll",
    "how'll",

    "what'd",
    "who'd",
    "where'd",
    "when'd",
    "why'd",
    "how'd",

    # informal/common forms
    "ain't",
}

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=False, default="gpt2")
    parser.add_argument("--data-path", "-dp", type=str, required=True)
    parser.add_argument("--load-from", "-lf", type=str, choices=["local", "hf"], required=True)
    parser.add_argument("--data-type", "-dt", type=str, default=None)
    parser.add_argument("--data-column", "-dc", type=str, choices=["text", "nonce", "core"], default="text")
    parser.add_argument("--split", "-sp", type=str, required=True, help="train/dev/test")
    parser.add_argument("--tokenize", "-t", action="store_true")
    parser.add_argument("--slice", "-s", action="store_true")
    parser.add_argument("--block-size", "-bs", type=int, required=True)
    parser.add_argument( "--kept-indices", "-ki", type=str, default=None, help="Path to json file")
    parser.add_argument( "--aoa", "-aoa", type=str, default=None, help="Path to aoa csv file")
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
        "--shuffle", action="store_true",
        help="Shuffle after slicing."
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


def tokenize_examples(examples, tokenizer, column_name: str, padding: bool, max_length: int, known_words: dict = {}):
    ensure_tokenizer_padding(tokenizer)
    if padding:
        result = tokenizer(
            examples[column_name],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            add_special_tokens=False,
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
            return_offsets_mapping=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        if tokenizer.eos_token_id is not None:
            result["input_ids"] = [
                ids + [tokenizer.eos_token_id]
                for ids in result["input_ids"]
            ]

            result["attention_mask"] = [
                mask + [1]
                for mask in result["attention_mask"]
            ]
        result["labels"] = [ids[:] for ids in result["input_ids"]]
    
    all_labels = []
    all_attention_masks = []
    texts = examples[column_name]
    for input_ids, attention_mask, offsets, text in zip(
        result["labels"],
        result["attention_mask"],
        result["offset_mapping"],
        texts 
    ):
        new_labels = input_ids.copy()
        new_attention_mask = attention_mask.copy()
        if not known_words:
            all_labels.append(new_labels)
            all_attention_masks.append(new_attention_mask)
            continue

        # 找出原文中的 word
        words = []
        word_spans = []

        for match in re.finditer(r"\S+", text):
            word = match.group()
            start = match.start()
            end = match.end()
            words.append(word)
            word_spans.append((start, end))

        for token_idx, (token_start, token_end) in enumerate(offsets):
            # special token 通常 offset=(0, 0)
            if token_start == token_end:
                continue

            # 找到这个 subword 属于哪个 word
            for word, (word_start, word_end) in zip(words, word_spans):
                overlap = (
                    token_start < word_end
                    and token_end > word_start
                )

                if overlap:
                    # stripped_word: str = re.sub(r'[^a-zA-Z]', '', word).lower()
                    stripped_word = word.strip(string.punctuation).lower()
                    token_text = text[token_start:token_end]
                    # check if the token contains any alphabetic characters
                    is_token_containing_alpha = re.search(r'[a-zA-Z]', token_text) is not None
                    # check if the stripped word is not empty and not in known_words
                    word_not_in_known_words = stripped_word and stripped_word not in known_words
                    if is_token_containing_alpha and word_not_in_known_words:
                        # token_str = tokenizer.decode([input_ids[token_idx]])
                        # print(f"token: '{token_str}', word: '{word}', stripped_word: '{stripped_word}' not in known_words")
                        new_labels[token_idx] = -100
                        new_attention_mask[token_idx] = 0
                    break

        all_labels.append(new_labels)
        all_attention_masks.append(new_attention_mask)

    result["labels"] = all_labels
    result["attention_mask"] = all_attention_masks
    del result["offset_mapping"]
    return result


def chunk_texts_to_blocks(examples, block_size: int):
    input_ids = list(chain.from_iterable(examples["input_ids"]))
    labels = list(chain.from_iterable(examples["labels"]))
    attention_mask = list(chain.from_iterable(examples["attention_mask"]))
    assert len(input_ids) == len(labels) == len(attention_mask)
    total_length = (len(input_ids) // block_size) * block_size

    input_blocks = [
        input_ids[i:i + block_size]
        for i in range(0, total_length, block_size)
    ]

    label_blocks = [
        labels[i:i + block_size]
        for i in range(0, total_length, block_size)
    ]

    attention_mask_blocks = [
        attention_mask[i:i + block_size]
        for i in range(0, total_length, block_size)
    ]

    return {
        "input_ids": input_blocks,
        "attention_mask": attention_mask_blocks,
        "labels": label_blocks,
    }


def main():
    args = read_args()
    validate_args(args)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(">>> Loading data ...")
    datasets = load_custom_dataset(args.data_path, args.data_type, args.load_from)
    if isinstance(datasets, dict):
        dataset = datasets[args.split]
    else:
        dataset = datasets
    print(f"  -> Dataset size: {len(dataset)}")

    kept_indices = None
    if args.kept_indices and Path(args.kept_indices).is_file():
        with open(args.kept_indices, "r") as f:
            kept_indices = json.load(f)
        print(f"  -> {len(kept_indices)} examples will be kept based on the provided indices.")
        print(">>> Filtering dataset based on kept indices ...")
        dataset = dataset.select(kept_indices)
        print(f"  -> Dataset size after filtering: {len(dataset)}")

    dataset = slice_dataset(dataset, args.start_from, args.data_limit)
    dataset = maybe_shuffle_dataset(dataset, shuffle=args.shuffle, seed=args.shuffle_seed)

    # === TOKENIZATION ===
    tokenized_dataset = dataset
    if args.tokenize:
        print(">>> Tokenizing ...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        # add special tokens if they are not already present
        special_tokens_dict = {}
        new_tokens_list = ["<UNK>", "<ENT>" ]
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = tokenizer.eos_token
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = tokenizer.eos_token
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = tokenizer.eos_token
        if special_tokens_dict:
            tokenizer.add_special_tokens(special_tokens_dict)
            print(f"  -> Added special tokens: {special_tokens_dict}")
        if new_tokens_list:
            tokenizer.add_tokens(new_tokens_list)
            print(f"  -> Added new tokens: {new_tokens_list}")

        padding = True
        if args.slice:
            padding = False
        
        known_words = {}
        if args.aoa:
            with open(args.aoa, "r") as f:
                known_words = load_aoa(f, AOA_THRESHOLD)
            for word in CONTRACTIONS:
                known_words[word.lower()] = "0.0"  # Add contractions to known words with a dummy AoA value

        map_func = partial(
            tokenize_examples,
            tokenizer=tokenizer,
            column_name=args.data_column,
            padding=padding,
            max_length=args.block_size,
            known_words=known_words
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
        map_func = partial(chunk_texts_to_blocks, block_size=args.block_size)

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
