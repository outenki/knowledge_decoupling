"""
Generate word frequency based on dictionary and corpus.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
import pandas as pd
import spacy
from tqdm import tqdm

DISABLED_PIPES = ["parser", "textcat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", nargs="?", default=".")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--model-name", default="en_core_web_sm")
    parser.add_argument(
        "--dataset-name",
        default="MAKILINGDING/english_dictionary",
    )
    return parser.parse_args()


def load_nlp(model_name: str):
    try:
        return spacy.load(model_name, disable=DISABLED_PIPES)
    except OSError:
        spacy.cli.download(model_name)
        return spacy.load(model_name, disable=DISABLED_PIPES)


def preprocess_text(text: str) -> str:
    return text.replace(".-", ". -")


def count_doc_tokens(doc) -> Counter:
    token_count: Counter = Counter()
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num:
            continue
        token_key = f"{token.lemma_.lower()}|{token.pos_}|{token.ent_type_}"
        token_count[token_key] += 1
    return token_count


def main() -> None:
    args = parse_args()
    nlp = load_nlp(args.model_name)
    ds = load_dataset(args.dataset_name)["train"]

    entry_num = len(ds)
    print(f"Total number of dictionary entries: {entry_num}")
    print(
        f"Using model={args.model_name}, "
        f"batch_size={args.batch_size}, n_process={args.n_process}"
    )

    definitions = (preprocess_text(item["definition"]) for item in ds)
    print("Processing definitions and counting token frequencies...")
    docs = nlp.pipe(
        definitions,
        batch_size=args.batch_size,
        n_process=args.n_process,
    )

    token_statistics = {}
    total_token_count = 0
    for doc in tqdm(docs, total=entry_num, desc="Counting tokens"):
        doc_token_count = count_doc_tokens(doc)
        for token, freq in doc_token_count.items():
            if token not in token_statistics:
                token_statistics[token] = {"count": 0, "entry_count": 0}
            token_statistics[token]["count"] += freq
            token_statistics[token]["entry_count"] += 1
            total_token_count += freq
    print(f"Total unique tokens: {len(token_statistics)}")
    print("Calculating mean and normalized frequencies...")
    for token, freq in token_statistics.items():
        freq["mean_frequency"] = freq["count"] / entry_num
        freq["normalized_frequency"] = freq["count"] / total_token_count

    output_path = Path(args.output_path) / args.model_name
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "token_frequency.json"
    print(f"Saving token frequency to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(token_statistics, f, indent=4)

    df = pd.DataFrame.from_dict(token_statistics, orient="index")
    df.index.name = "token"
    csv_output_file = output_file.with_suffix(".csv")
    print(f"Saving token frequency to {csv_output_file}...")
    df.to_csv(csv_output_file)


if __name__ == "__main__":
    main()
