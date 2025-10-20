# merge pre-generated lemma_blacklists and nonce_word_bank.json

from pathlib import Path
import argparse
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path-to-lemma-blacklists', '-lb', dest='path_lemma_blacklists', type=str,
        help='Path to lemma_blacklists. Files are read in a recursed way.'
    )
    parser.add_argument(
        '--path-to-nonce-word-banks', '-wb', dest='path_word_banks', type=str,
        help='Path to word_banks. Files are read in a recursed way.'
    )
    parser.add_argument(
        '--output-path', '-o', dest='output_path', required=True, type=str,
        help='Path to save output. '
    )
    return parser.parse_args()


def merge_lemma_blacklists(blacklist: set[str], path_to_new_blacklist: str | Path) -> set:
    with open(path_to_new_blacklist, "r") as f:
        for line in f:
            blacklist.add(line.strip())
    return blacklist


def load_word_bank(path_to_word_bank: str | Path) -> dict:
    with open(path_to_word_bank, "r") as f:
        try:
            word_bank = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to load {path_to_word_bank}. Skipping...")
            return {}
    return {k: set([tuple(t) for t in v]) for k, v in word_bank.items()}


def merge_word_bank(word_bank: dict, path_to_new_word_bank: str | Path) -> dict:
    new_word_bank = load_word_bank(path_to_new_word_bank)
    for k, v in new_word_bank.items():
        if k in word_bank:
            word_bank[k] = word_bank[k] | v
        else:
            word_bank[k] = v
    return word_bank


def main():
    args = parse_args()
    path_lemma_blacklists = []
    path_word_banks = []
    if args.path_lemma_blacklists:
        path_lemma_blacklists = list(Path(args.path_lemma_blacklists).rglob("lemma_blacklist"))
    if args.path_word_banks:
        path_word_banks = list(Path(args.path_word_banks).rglob("nonce_word_bank.json"))

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    lemma_blacklist = set()
    output_lemma_blacklist = output_path / "lemma_blacklist"
    if output_lemma_blacklist.exists():
        with open(output_lemma_blacklist, "r") as f:
            lemma_blacklist = set([line.strip() for line in f if line])

    word_bank = {}
    output_word_bank = output_path / "nonce_word_bank.json"
    if output_word_bank.exists():
        word_bank = load_word_bank(output_word_bank)

    for p in tqdm(path_lemma_blacklists, desc="Merging lemma_blacklists"):
        lemma_blacklist = merge_lemma_blacklists(lemma_blacklist, p)
    for p in tqdm(path_word_banks, desc="Merging nonce_word_banks"):
        word_bank = merge_word_bank(word_bank, p)

    if lemma_blacklist:
        print(f"Saving merged lemma_blacklist with {len(lemma_blacklist)} entries to {output_lemma_blacklist}")
        with open(output_lemma_blacklist, "w") as f:
            for lemma in lemma_blacklist:
                f.write(f"{lemma}\n")

    if word_bank:
        print(f"Saving merged nonce_word_bank with {len(word_bank)} entries to {output_word_bank}")
        word_bank = {k: [list(t) for t in v] for k, v in word_bank.items()}
        with open(output_word_bank, "w") as f:
            json.dump(word_bank, f, indent=4)


main()
