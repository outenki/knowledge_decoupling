import argparse
import json
from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset, Dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str, required=True,
        help='Name of the dataset to load from Hugging Face'
    )
    parser.add_argument(
        '--subset-name', '-sn', dest='subset_name', type=str, required=True,
        help='Name of the sebset'
    )
    parser.add_argument(
        '--output-path', '-o', dest='output_path', type=str, required=True,
        help='Path to save results.'
    )
    return parser.parse_args()


def generate_qa_data(dataset: Dataset) -> dict:
    qa_data = []
    for item in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
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


args = read_args()
print(f"making dirs: {args.output_path}")
Path(args.output_path).mkdir(parents=True, exist_ok=True)


dataset_dict = load_dataset(args.data_name, args.subset_name)
for split, dataset in dataset_dict.items():
    print(f"Processing split: {split} with {len(dataset)} samples")
    qa_data = generate_qa_data(dataset)
    output_file = Path(args.output_path) / f"{split}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)
    print(f"Saved {len(qa_data)} samples to {output_file}")