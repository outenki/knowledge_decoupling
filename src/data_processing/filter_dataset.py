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
from src.lib.dataset import slice_dataset


DEFAULT_MODEL_NAME = "en_core_web_sm"
DISABLED_PIPES = ["parser", "textcat"]
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_THRESHOLD = 1e-6
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_PROC = min(4, multiprocessing.cpu_count())
NLP = None
NLP_SENTENCE_SEGMENTER = None
SKIP_SOURCES = {"finemath", "stack_edu", "infimm_webmath", "stack-edu", "infimm-webmath"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--token-frequency-path", required=True, type=str)
    parser.add_argument("--keep-oov", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--start-from", required=True, type=int)
    parser.add_argument("--data-limit", type=int, default=0)
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


def load_nlp_sentence_segmenter(model_name: str):
    global NLP_SENTENCE_SEGMENTER
    if NLP_SENTENCE_SEGMENTER is None:
        try:
            NLP_SENTENCE_SEGMENTER = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)
            NLP_SENTENCE_SEGMENTER = spacy.load(model_name)
        NLP_SENTENCE_SEGMENTER.add_pipe('sentencizer')
    return NLP_SENTENCE_SEGMENTER


def preprocess_text(text: str) -> str:
    return text.replace(".-", ". -")


def build_allowed_token_keys(token_frequencies: dict, filter_key: str, threshold: float) -> set[str]:
    return {
        token_key
        for token_key, frequency_info in token_frequencies.items()
        if frequency_info.get(filter_key, 0) >= threshold and token_key.split("|")[0].isalpha()
    }

def build_allowed_token_candidates(allowed_token_keys: set[str]) -> dict[tuple[str, str], list[str]]:
    candidates: dict[tuple[str, str], list[str]] = {}
    for token_key in allowed_token_keys:
        lemma, pos, ent_type = token_key.split("|", 2)
        candidates.setdefault((pos, ent_type), []).append(lemma)
        candidates.setdefault((pos, ""), []).append(lemma)
    return candidates


def process_sent(
    sent,
    allowed_token_keys: set[str],
    all_tokens_keys: dict,
    keep_oov: bool,
    replacement_candidates: dict[tuple[str, str], list[str]],
) -> dict:
    """
    Process document in a single pass: check if simplification is needed,
    and simplify tokens while preserving whitespace. Returns filtered result.
    """
    nlp = load_nlp(DEFAULT_MODEL_NAME)
    doc = nlp(sent.text)
    if doc.text.strip() == "":
        return None

    pieces = []
    has_disallowed_token = False
    needs_replacement = False

    for token in doc:
        # if token.is_stop or token.is_punct or token.like_num or token.is_space or token.pos_ in {"NUM", "PUNCT", "INTJ", "SYM", "X", "CCONJ", "ADP", "DET", "PRON", "SCONJ"}:
        if token.is_stop or token.is_punct or token.like_num or token.is_space or token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}:
            pieces.append(token.text_with_ws)
            continue

        token_key = f"{token.lemma_.lower()}|{token.pos_}|{token.ent_type_}"
        
        if token_key in allowed_token_keys:
            pieces.append(token.text_with_ws)
        elif token_key not in all_tokens_keys and keep_oov:
            pieces.append(token.text_with_ws)
        else:
            has_disallowed_token = True
            needs_replacement = True
            candidates = replacement_candidates.get((token.pos_, token.ent_type_), [])
            if not candidates:
                candidates = replacement_candidates.get((token.pos_, ""), [])
            if not candidates:
                # Cannot replace this token, must drop
                return {"text": doc.text, "source": "drop"}
            candidates = random.sample(candidates, min(20, len(candidates)))  # sample a few candidates to try
            candidate_inflected = None
            for candidate_lemma in candidates:
                candidate_inflected = inflect_candidate(candidate_lemma, token.tag_)
                if candidate_inflected is not None:
                    break
            if candidate_inflected is None:
                # Cannot replace this token, must drop
                return {"text": doc.text, "source": "drop"}
            pieces.append(candidate_inflected + token.whitespace_)

    if not has_disallowed_token:
        return {"text": doc.text, "source": "keep"}
    
    if needs_replacement:
        return {"text": "".join(pieces), "source": "replaced"}
    
    return {"text": doc.text, "source": "drop"}


def process_document(
    doc,
    allowed_token_keys: set[str],
    all_tokens_keys: dict,
    keep_oov: bool,
    replacement_candidates: dict[tuple[str, str], list[str]],
) -> dict:
    sents = doc.sents
    source = "keep"
    text = ""
    for sent in sents:
        result = process_sent(
            sent,
            allowed_token_keys=allowed_token_keys,
            all_tokens_keys=all_tokens_keys,
            keep_oov=keep_oov,
            replacement_candidates=replacement_candidates,
        )
        if not result:
            continue
        text = text + result["text"]
        if result["source"] == "drop":
            source = "drop"
            return {"text": text, "source": source}
        elif source != "drop" and result["source"] == "replaced":
            source = "replaced"
    return {"text": text, "source": source}


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
    nlp_sent = load_nlp_sentence_segmenter(model_name)
    replacement_candidates = build_allowed_token_candidates(allowed_token_keys)

    def iter_filtered_examples():
        example_stream = (
            (preprocess_text(example[text_column]), dict(example))
            for example in ds
        )
        _id = 0
        docs = nlp_sent.pipe(
            example_stream,
            as_tuples=True,
            batch_size=batch_size,
            n_process=num_proc,
            disable=[p for p in nlp_sent.pipe_names if p != "sentencizer"],
        )

        for doc, example in tqdm(
            docs,
            total=len(ds),
            desc="Filtering dataset based on token frequencies",
        ):
            _id += 1
            result = process_document(
                doc,
                allowed_token_keys=allowed_token_keys,
                all_tokens_keys=all_tokens_keys,
                keep_oov=keep_oov,
                replacement_candidates=replacement_candidates,
            )
            if result is None:
                continue

            example[text_column] = result["text"]
            example["ori_text"] = doc.text if result["source"] == "replaced" else None
            example["filter_source"] = result["source"]
            example["id"] = _id
            yield example

    return Dataset.from_generator(iter_filtered_examples)


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
    print(f"Allowed token keys: {len(allowed_token_keys)}/{len(token_frequencies)}")

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
        print(f"Filtering dataset by skipping sources: {SKIP_SOURCES} ...")
        kept_indices = drop_skipped_sources(ds, SKIP_SOURCES)
        # save kept indices for reproducibility and debugging
        Path(args.kept_indices_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving kept indices after source filter to {args.kept_indices_path} ...")
        with open(args.kept_indices_path, "w") as f:
            json.dump(kept_indices, f)
    if len(kept_indices) < len(ds):
        print(f"Dropping {len(ds) - len(kept_indices)} ({len(kept_indices)} left) examples from skipped sources: {SKIP_SOURCES}")
        ds = ds.select(kept_indices)

    # Apply data limit if specified
    if args.data_limit > 0:
        ds = slice_dataset(ds, args.start_from, args.data_limit)

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

    # count kept samples where filter_source is not "drop"
    print(f"Calculating kept samples")
    kept_ds = filtered_ds.filter(lambda x: x["filter_source"] != "drop")
    print(f"Kept {len(kept_ds)} / {len(ds)} examples")
    filtered_ds.save_to_disk(output_path)
    print(f"Filtered dataset saved to {output_path}")

    # save first 10 samples as json file for checking
    filtered_ds.select(range(10)).to_json(Path(output_path) / "examples.json")


if __name__ == "__main__":
    main()
