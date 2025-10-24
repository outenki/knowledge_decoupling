import argparse
import json
from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset, Dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str, required=True,
        choices=['ai2_arc', 'boolq', 'boolq_psg', 'qasc'],
        help='Name of the dataset to load from Hugging Face'
    )
    parser.add_argument(
        '--subset-name', '-sn', dest='subset_name', type=str, required=False,
        default=None,
        help='Name of the subset'
    )
    parser.add_argument(
        '--output-path', '-o', dest='output_path', type=str, required=True,
        help='Path to save results.'
    )
    return parser.parse_args()


def generate_qa_data_from_ai2_arc(dataset: Dataset) -> list[dict]:
    qa_data = []
    for item in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(item, dict)
        prompt = item["question"]
        options = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        answer = options[labels.index(answer_key)]
        qa_data.append({
            "text": prompt + " " + answer,
            "prompt": prompt,
            "options": options,
            "answer": answer
        })
    return qa_data


def generate_qa_data_from_boolq(dataset: Dataset, passage=False) -> list[dict]:
    qa_data = []
    for item in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(item, dict)
        prompt = item["question"] + "?"
        if passage:
            prompt = item["passage"] + " " + prompt
        answer = "Yes" if item["answer"] else "No"
        qa_data.append({
            "text": prompt + " " + answer,
            "prompt": prompt,
            "options": ["Yes", "No"],
            "answer": answer
        })
    return qa_data


def generate_qa_data_from_qasc(dataset: Dataset) -> list[dict]:
    qa_data = []
    for item in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(item, dict)
        prompt = item["question"]
        options = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        if answer_key not in labels:
            continue  # Skip if the answer key is not in the labels
        answer = options[labels.index(answer_key)]
        qa_data.append({
            "text": prompt + " " + answer,
            "prompt": prompt,
            "options": options,
            "answer": answer
        })
    return qa_data


args = read_args()
print(vars(args))
print(f"making dirs: {args.output_path}")
Path(args.output_path).mkdir(parents=True, exist_ok=True)

if args.data_name == "boolq_psg":
    dataset_dict = load_dataset("boolq", args.subset_name)
else:
    dataset_dict = load_dataset(args.data_name, args.subset_name)
assert isinstance(dataset_dict, dict)
for split, dataset in dataset_dict.items():
    print(f"Processing split: {split} with {len(dataset)} samples")
    if args.data_name == "ai2_arc":
        qa_data = generate_qa_data_from_ai2_arc(dataset)
    elif args.data_name == "boolq":
        qa_data = generate_qa_data_from_boolq(dataset)
    elif args.data_name == "boolq_psg":
        qa_data = generate_qa_data_from_boolq(dataset, passage=True)
    elif args.data_name == "qasc":
        qa_data = generate_qa_data_from_qasc(dataset)
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}")
    output_file = Path(args.output_path) / f"{split}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)
    print(f"Saved {len(qa_data)} samples to {output_file}")
