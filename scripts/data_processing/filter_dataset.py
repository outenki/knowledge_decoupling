"""Filter or rewrite dataset examples based on allowed token keys."""

import argparse
import json
import multiprocessing
from pathlib import Path
import random

from datasets import Dataset
from datasets import load_dataset
import spacy
from tqdm.auto import tqdm

from src.lib.dataset import drop_skipped_sources
from src.lib.text import inflect_candidate


DEFAULT_MODEL_NAME = "en_core_web_sm"
DISABLED_PIPES = ["parser", "textcat"]
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_THRESHOLD = 1e-6
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_PROC = min(4, multiprocessing.cpu_count())
NLP = None
SKIP_SOURCES = {"finemath", "stack-edu", "infimm-webmath"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--token-frequency-path", required=True, type=str)
    parser.add_argument("--keep-oov", action="store_true")
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

def build_allowed_token_candidates(allowed_token_keys: set[str]) -> dict[tuple[str, str], list[str]]:
    candidates: dict[tuple[str, str], list[str]] = {}
    for token_key in allowed_token_keys:
        lemma, pos, ent_type = token_key.split("|", 2)
        candidates.setdefault((pos, ent_type), []).append(lemma)
    return candidates


def simplify_text(
    doc,
    allowed_token_keys: set[str],
    all_tokens_keys: dict,
    keep_oov: bool,
    replacement_candidates: dict[tuple[str, str], list[str]],
) -> str | None:
    """
    Replace unsupported tokens token-by-token while preserving whitespace.
    """
    pieces = []
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num or token.is_space:
            pieces.append(token.text_with_ws)
            continue

        token_key = f"{token.lemma_.lower()}|{token.pos_}|{token.ent_type_}"
        if token_key in allowed_token_keys:
            pieces.append(token.text_with_ws)
        elif token_key not in all_tokens_keys and keep_oov:
            pieces.append(token.text_with_ws)
        else:
            candidates = replacement_candidates.get((token.pos_, token.ent_type_), [])
            if not candidates:
                return None
            candidate_lemma = random.choice(candidates)
            candidate_inflected = inflect_candidate(candidate_lemma, token.tag_)
            replacement = candidate_inflected or candidate_lemma
            pieces.append(replacement + token.whitespace_)
    return "".join(pieces)


def keep_example(
    doc,
    allowed_token_keys: set[str],
    all_tokens_keys: dict,
    keep_oov: bool,
    replacement_candidates: dict[tuple[str, str], list[str]],
) -> dict | None:
    if doc.text.strip() == "":
        return None

    has_disallowed_token = False
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num or token.is_space:
            continue
        token_key = f"{token.lemma_.lower()}|{token.pos_}|{token.ent_type_}"
        if token_key not in all_tokens_keys and keep_oov:
            continue
        if token_key not in allowed_token_keys:
            has_disallowed_token = True

    if not has_disallowed_token:
        return {"text": doc.text, "source": "keep"}

    simplified_text = simplify_text(
        doc,
        allowed_token_keys=allowed_token_keys,
        all_tokens_keys=all_tokens_keys,
        keep_oov=keep_oov,
        replacement_candidates=replacement_candidates,
    )
    if simplified_text is not None:
        return {"text": simplified_text, "source": "replaced"}
    return {"text": doc.text, "source": "drop"}


def filter_dataset(
    ds: Dataset,
    text_column: str,
    allowed_token_keys: set[str],
    all_tokens_keys: dict,
    batch_size: int,
    model_name: str,
    num_proc: int,
    keep_oov: bool,
) -> Dataset:
    nlp = load_nlp(model_name)
    replacement_candidates = build_allowed_token_candidates(allowed_token_keys)

    def iter_filtered_examples():
        example_stream = (
            (preprocess_text(example[text_column]), dict(example))
            for example in ds
        )
        docs = nlp.pipe(
            example_stream,
            as_tuples=True,
            batch_size=batch_size,
            n_process=num_proc,
        )

        for doc, example in tqdm(
            docs,
            total=len(ds),
            desc="Filtering dataset based on token frequencies",
        ):
            result = keep_example(
                doc,
                allowed_token_keys=allowed_token_keys,
                all_tokens_keys=all_tokens_keys,
                keep_oov=keep_oov,
                replacement_candidates=replacement_candidates,
            )
            if result is None:
                continue

            example[text_column] = result["text"]
            example["filter_source"] = result["source"]
            yield example

    return Dataset.from_generator(iter_filtered_examples)


# def iter_kept_indices(
#     ds,
#     text_column: str,
#     allowed_token_keys: set[str],
#     all_tokens_keys: dict,
#     batch_size: int,
#     model_name: str,
#     num_proc: int,
#     keep_oov: bool,
#     replace: bool,
# ):
#     nlp = load_nlp(model_name)
#     total_examples = len(ds)
#     texts = (preprocess_text(example[text_column]) for example in ds)
#     docs = nlp.pipe(texts, batch_size=batch_size, n_process=num_proc)

#     for index, doc in enumerate(
#         tqdm(docs, total=total_examples, desc="Filtering dataset based on token frequencies")
#     ):
#         yield keep_example(doc, allowed_token_keys, all_tokens_keys):


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
        kept_indices = drop_skipped_sources(ds, SKIP_SOURCES)
        # save kept indices for reproducibility and debugging
        Path(args.kept_indices_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.kept_indices_path, "w") as f:
            json.dump(kept_indices, f)
    if len(kept_indices) < len(ds):
        print(f"Dropping {len(ds) - len(kept_indices)} examples from skipped sources: {SKIP_SOURCES}")
        ds = ds.select(kept_indices)

    print(f"Total examples in the dataset: {len(ds)}")
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


    filtered_ds = filter_dataset(
        ds=ds,
        text_column=args.text_column,
        allowed_token_keys=allowed_token_keys,
        all_tokens_keys=token_frequencies,
        batch_size=args.batch_size,
        model_name=args.model_name,
        num_proc=args.num_proc,
        keep_oov=args.keep_oov,
    )

    print(f"Kept {len(filtered_ds)} / {len(ds)} examples")
    output_dir = output_path / "filtered_dataset"
    filtered_ds.save_to_disk(output_dir)
    print(f"Filtered dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
