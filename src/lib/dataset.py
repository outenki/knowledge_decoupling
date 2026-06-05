from typing import Any
from pathlib import Path
from functools import partial
from tqdm import tqdm
import json

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset, load_from_disk
from datasets import concatenate_datasets

from src.lib.text import split_text_to_sentences


def load_custom_dataset(data_name: str, data_type: str | None, load_from: str) -> Dataset | DatasetDict | Any:
    # Load dataset from local path with default type
    if load_from == "local" and data_type is None:
        print(f"Loading dataset {data_name} from local disk...")
        full_data_path = Path(data_name)
        return load_from_disk(full_data_path)

    # Load dataset from local path with specific type
    elif load_from == "local" and data_type is not None:
        print(f"Loading dataset {data_name} from local disk with type {data_type}...")
        full_data_path = str(data_name)
        return load_dataset(data_type, data_files=full_data_path)

    # Load dataset from Hugging Face
    elif load_from == "hf":
        print(f"Loading dataset {data_name} from Hugging Face...")
        if data_name.lower() == "wikimedia":
            return load_dataset("wikimedia/wikipedia", "20231101.en")
        if data_name.lower() == "wikipedia":
            return load_dataset("wikipedia", "20220301.en")
        if data_name.lower() == "wikitext-103":
            return load_dataset("wikitext", "wikitext-103-v1")
        if data_name.lower() == "wikitext-2":
            return load_dataset("wikitext", "wikitext-2-v1")
        if data_name.lower() == "smollm2-10B".lower():
            return load_dataset("EleutherAI/SmolLM2-135M-10B")
        if data_name.lower() == "smollm2-20B".lower():
            return load_dataset("EleutherAI/SmolLM2-135M-20B")
        else:
            return load_dataset(data_name)
    else:
        raise ValueError("Invalid load_from option.")


def load_texts_from_dataset_batch(dataset: Dataset, batch_idx: int, batch_size: int) -> list[str]:
    """
    Loads texts from a Hugging Face Dataset object.

    Args:
        dataset (Dataset): A Hugging Face Dataset object.

    Returns:
        list[str]: A list of texts from the dataset.
    """
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(dataset))
    if start >= len(dataset):
        return []
    rng = range(start, end)
    if isinstance(dataset, Dataset):
        return dataset.select(rng)["text"]
    else:
        raise ValueError("Unsupported dataset type. Please provide a Hugging Face Dataset object.")


def skip_dataset_by_column(dataset: Dataset, column_name, column_values):
    print(f"**** Skipping dataset where column_name == {column_name} and values in [{column_values}]...")
    size_before = len(dataset)
    dataset = dataset.filter(lambda example: column_name in example and example[column_name] not in set(column_values))
    size_after = len(dataset)
    print(f"**** Datasize {size_before} -> {size_after} after skipping ...")
    return dataset


def slice_dataset(dataset: Dataset, start: int, limit: int) -> Dataset:
    print(f"**** Slicing dataset from index {start} with limit {limit} ...")
    assert start <= len(dataset), f"start index {start} is out of bounds for dataset of size {len(dataset)}"
    if start == 0 and limit <= 0:
        print(f"**** {len(dataset)} samples selected...")
        return dataset
    end = start + limit
    if end <= start or end >= len(dataset):
        end = len(dataset)
    print(f"**** {end - start} samples selected...")
    return dataset.select(range(start, end))


def split_column_to_sents(examples, column: str):
    # 返回新的列 "sentences"，每条文本拆成句子
    all_sents = []
    for text in examples[column]:
        all_sents.extend(split_text_to_sentences(text))
    return {column: all_sents}


def simple_split_to_sents(dataset: Dataset, column, num_proc=1, batch_size=1000) -> Dataset:
    map_split_func = partial(split_column_to_sents, column=column)
    print("Splitting dataset to sentences")
    dataset = dataset.map(
        map_split_func,
        num_proc=num_proc,
        batch_size=batch_size,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Splitting data to sentences"
    )
    print(f"Output size of splitting: {len(dataset)}")
    return dataset


def drop_skipped_sources(ds: Dataset, skip_sources):
    if "source" not in ds.column_names:
        return ds
    sources = ds["source"]
    kept_indices = [
        index for index, source in tqdm(enumerate(sources), desc="Dropping skipped sources", total=len(sources)) if source not in skip_sources
    ]
    if len(kept_indices) == len(ds):
        return ds
    return kept_indices


def format_qa_prompt(example):
    prompt = ""
    context = example.get("context", "").strip()
    if context:
        prompt += "Context:\n" + context + "\n\n" 

    question = example["question"]
    prompt += "Question: " + question + "\n\n"

    options = ""
    if "options" in example:
        options = "\n".join(example["options"]).strip()
    if options:
        prompt += "Options:\n" + options + "\n\n"

    prompt += "Answer:\n"
    response = example["answer"]
    return prompt, response


def generate_qa_message(example):
    question = example["question"]
    context = example.get("context", "").strip()
    options = ""
    if "options" in example:
        options = "\n".join(example["options"]).strip()
    answer = example["answer"]

    content = f"Question:\n{question}\n\n"
    if context:
        content += f"Context:\n{context}\n\n"
    if options and options:
        content += f"Options:\n{options}\n\n"
    if context and not options:
        content += "Answer the question based on the context."
    if not context and options:
        content += "Answer the question using the exact text from the options."
    if context and options:
        content += "Answer the question using the exact text from the options based on the context."
    if not context and not options:
        content += "Answer the question."

    return [
        {
            "role": "user",
            "content": content
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]


def select_data_by_indices(dataset: Dataset, indices_fn: str) -> Dataset:
    kept_indices = None
    if not Path(indices_fn).is_file():
        print(f"Warning: kept indices file not found at {indices_fn}. All examples will be kept.")
    else:
        with open(indices_fn, "r") as f:
            kept_indices = json.load(f)
        percent = (len(kept_indices) / len(dataset)) * 100 if len(dataset) else 0
        print(f"  -> {len(kept_indices)} ({percent:.2f}%) examples will be kept based on the provided indices.")
        print(">>> Filtering dataset based on kept indices ...")
        dataset = dataset.select(kept_indices)
        print(f"  -> Dataset size after filtering: {len(dataset)}")
    return dataset


def _limit_dataset_dict(data_dict: DatasetDict, data_limit: int) -> DatasetDict:
    for k, dt in data_dict.items():
        dt = dt.shuffle(seed=42)
        if k == "train":
            data_dict[k] = slice_dataset(dt, 0, data_limit)
        else:
            # For validation and test sets, use a smaller limit
            _data_limit = data_limit // 10
            data_dict[k] = slice_dataset(dt, 0, _data_limit)
    return data_dict


def _load_dataset_dict(data_path: str) -> DatasetDict:
    dataset = load_custom_dataset(data_path, None, "local")
    if isinstance(dataset, Dataset):
        data_dict = DatasetDict({"train": dataset})
    elif isinstance(dataset, DatasetDict):
        data_dict = dataset
    else:
        raise TypeError(f"Invalid data type {type(dataset)}")
    return data_dict

def load_and_limit_dataset_dict(data_path: str, data_limit: int) -> DatasetDict:
    """
    Load data and limit the number of samples for training.
    For validation and test sets, use a smaller limit (data_limit // 10).
    """
    print(f">>> Loading dataset from {data_path}")
    _data_dict = _load_dataset_dict(data_path)
    print(_data_dict)
    print(">>> data loaded")
    if data_limit > 0:
        _data_dict = _limit_dataset_dict(_data_dict, data_limit)
    return _data_dict


def load_dataset_for_training(data_paths: list, data_limits: list) -> tuple[Dataset, Dataset]:
    data_list = []
    if not data_limits or len(data_limits) == 0:
        data_limits = [0] * len(data_paths)
    assert len(data_paths) == len(data_limits), "data_paths and data_limits should have the same length."
    for dp, dl in zip(data_paths, data_limits):
        _data_dict = load_and_limit_dataset_dict(dp, dl)
        data_list.append(_data_dict)
    data_dict = DatasetDict({
        split: concatenate_datasets([dd[split] for dd in data_list])
        for split in data_list[0].keys()
    })
    print(">>> Dataset:", data_dict)

    train_dataset: Dataset | None = None
    eval_dataset: Dataset | None = None
    train_dataset = data_dict["train"]
    for split in ("val", "validation", "dev"):
        if split in data_dict:
            eval_dataset = data_dict[split]
            break
    if eval_dataset is None:
        data_dict = train_dataset .train_test_split(test_size=0.05, shuffle=True, seed=42)
        train_dataset = data_dict["train"]
        eval_dataset = data_dict["test"]
    print(">>> dataset features:", train_dataset.features)
    print(f">>> Training data size: {len(train_dataset)}")
    print(f">>> Eval data size: {len(eval_dataset)}")
    return train_dataset, eval_dataset
