from typing import Any
from pathlib import Path
from functools import partial

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset, load_from_disk
from lib.text import split_text_to_sentences


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
        if data_name.lower() == "smollm2":
            return load_dataset("EleutherAI/SmolLM2-135M-10B")
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
    assert start <= len(dataset)
    end = start + limit
    if end <= start or end >= len(dataset):
        end = len(dataset)
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
