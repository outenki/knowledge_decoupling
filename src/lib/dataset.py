from typing import Any
from pathlib import Path
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset, load_from_disk


def load_custom_dataset(data_path: str, data_name: str, data_type: str, load_from: str) -> Dataset | DatasetDict | Any:
    # Load dataset from local path with default type
    if load_from == "local" and data_type is None:
        print(f"Loading dataset {data_path}/{data_name} from local disk...")
        full_data_path = Path(data_path) / data_name
        return load_from_disk(full_data_path)

    # Load dataset from local path with specific type
    elif load_from == "local" and data_type is not None:
        print(f"Loading dataset {data_path}/{data_name} from local disk with type {data_type}...")
        full_data_path = str(Path(data_path) / data_name)
        return load_dataset(data_type, data_files=full_data_path)

    # Load dataset from Hugging Face
    elif load_from == "hf":
        print(f"Loading dataset {data_path}/{data_name} from Hugging Face...")
        print(f"load_dataset('{data_path}', '{data_name}')")
        return load_dataset(data_path, data_name)
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
