"""Filter a dataset by removing examples that contain very low-frequency tokens."""

import argparse
import json
import multiprocessing
from pathlib import Path

from datasets import load_dataset
import spacy
from tqdm.auto import tqdm


DEFAULT_MODEL_NAME = "en_core_web_sm"
DISABLED_PIPES = ["parser", "textcat"]
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_THRESHOLD = 1e-6
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_PROC = min(4, multiprocessing.cpu_count())
NLP = None
SKIP_OOV_TOKENS = True  # If True, we will keep examples with OOV tokens. Otherwise, we will filter them out.
SKIP_SOURCES = {"finemath", "stack-edu", "infimm-webmath"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--token-frequency-path", required=True, type=str)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--kept-indices-path", required=True, type=str)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--filter-key", required=True, type=str)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-proc", type=int, default=DEFAULT_NUM_PROC)
    return parser.parse_args()


def load_nlp(model_name: str):
    global NLP
    if NLP is None:
        try:
            NLP = spacy.load(model_name, disable=DISABLED_PIPES)
        except OSError:
            spacy.cli.download(model_name)
            NLP = spacy.load(model_name, disable=DISABLED_PIPES)
    return NLP


def preprocess_text(text: str) -> str:
    return text.replace(".-", ". -")


def build_allowed_token_keys(token_frequencies: dict, filter_key: str, threshold: float) -> set[str]:
    return {
        token_key
        for token_key, frequency_info in token_frequencies.items()
        if frequency_info.get(filter_key, 0) >= threshold
    }


def keep_example(doc, allowed_token_keys: set[str], all_tokens_keys: dict) -> bool:
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num or token.is_space:
            continue
        token_key = f"{token.lemma_.lower()}|{token.pos_}|{token.ent_type_}"
        # deal with oov tokens by checking if the token key exists in the frequency dict. If not, we keep the example.
        if token_key not in all_tokens_keys:
            if token.ent_type_ or SKIP_OOV_TOKENS:
                continue
        if token_key not in allowed_token_keys:
            return False
    return True


def iter_kept_indices(
    ds,
    text_column: str,
    allowed_token_keys: set[str],
    all_tokens_keys: dict,
    batch_size: int,
    model_name: str,
    num_proc: int,
):
    nlp = load_nlp(model_name)
    total_examples = len(ds)
    texts = (preprocess_text(example[text_column]) for example in ds)
    docs = nlp.pipe(texts, batch_size=batch_size, n_process=num_proc)

    for index, doc in enumerate(
        tqdm(docs, total=total_examples, desc="Filtering dataset based on token frequencies")
    ):
        if keep_example(doc, allowed_token_keys, all_tokens_keys):
            yield index


def drop_skipped_sources(ds):
    if "source" not in ds.column_names:
        return ds

    sources = ds["source"]
    kept_indices = [index for index, source in tqdm(enumerate(sources), desc="Dropping skipped sources", total=len(sources)) if source not in SKIP_SOURCES]
    if len(kept_indices) == len(ds):
        return ds
    return kept_indices


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(args.token_frequency_path, "r") as f:
        token_frequencies = json.load(f)
    allowed_token_keys = build_allowed_token_keys(
        token_frequencies,
        args.filter_key,
        args.threshold,
    )

    print(f"Loading dataset {args.dataset_name}:{args.dataset_split}...")
    ds = load_dataset(args.dataset_name)[args.dataset_split]

    # Skip sources
    kept_indices = {}
    print(f"Trying to load kept indices after source filter from {args.kept_indices_path}...")
    if Path(args.kept_indices_path).exists():
        print(f"Loading kept indices after source filter from {args.kept_indices_path}...")
        with open(args.kept_indices_path, "r") as f:
            kept_indices = json.load(f)
    else:
        kept_indices = drop_skipped_sources(ds)
        # save kept indices for reproducibility and debugging
        Path(args.kept_indices_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.kept_indices_path, "w") as f:
            json.dump(kept_indices, f)
    if len(kept_indices) < len(ds):
        print(f"Dropping {len(ds) - len(kept_indices)} examples from skipped sources: {SKIP_SOURCES}")
        ds = ds.select(kept_indices)

    print(f"Total examples in the dataset: {len(ds)}")
    # limit the dataset for testing
    ds = ds.select(range(10000))
    if args.text_column not in ds.column_names:
        raise ValueError(
            f"Column '{args.text_column}' not found. Available columns: {ds.column_names}"
        )

    print(f"Loaded {len(ds)} examples from {args.dataset_name}:{args.dataset_split}")
    print(
        f"Filtering with threshold={args.threshold}, "
        f"batch_size={args.batch_size}, num_proc={args.num_proc}"
    )
    print(f"Allowed token keys: {len(allowed_token_keys)}")
    if "trf" in args.model_name and args.num_proc > 1:
        print(
            "Warning: transformer spaCy models often scale poorly with n_process > 1. "
            "This parallel path is primarily intended for en_core_web_sm/md."
        )

    kept_indices = list(
        iter_kept_indices(
            ds=ds,
            text_column=args.text_column,
            allowed_token_keys=allowed_token_keys,
            all_tokens_keys=token_frequencies,
            batch_size=args.batch_size,
            model_name=args.model_name,
            num_proc=args.num_proc,
        )
    )
    filtered_ds = ds.select(kept_indices)

    print(f"Kept {len(filtered_ds)} / {len(ds)} examples")
    output_dir = output_path / "filtered_dataset"
    filtered_ds.save_to_disk(output_dir)
    print(f"Filtered dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
