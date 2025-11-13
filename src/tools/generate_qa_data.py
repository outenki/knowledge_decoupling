import argparse
import json
from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset, Dataset


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str, required=True,
        choices=['ai2_arc', 'boolq', 'boolq_ctxt', 'qasc', "squad_v2", "squad_v2_ctxt"],
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
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        prompt = sample["question"]
        if not prompt.endswith("?"):
            prompt += "?"
        options = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        answer = options[labels.index(answer_key)]
        qa_data.append({
            "context": "",
            "text": prompt + " " + answer,
            "prompt": prompt,
            "options": options,
            "answer": answer
        })
    return qa_data


def generate_qa_data_from_boolq(dataset: Dataset, context=False) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        if not prompt.endswith("?"):
            prompt += "?"
        if context:
            prompt = sample["passage"] + " " + prompt
        answer = "Yes" if sample["answer"] else "No"
        qa_data.append({
            "context": sample["passage"],
            "text": prompt + " " + answer,
            "prompt": prompt,
            "options": ["Yes", "No"],
            "answer": answer
        })
    return qa_data


def generate_qa_data_from_qasc(dataset: Dataset) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        prompt = sample["question"]
        if not prompt.endswith("?"):
            prompt += "?"
        options = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        if answer_key not in labels:
            continue  # Skip if the answer key is not in the labels
        answer = options[labels.index(answer_key)]
        qa_data.append({
            "context": "",
            "text": prompt + " " + answer,
            "prompt": prompt,
            "options": options,
            "answer": answer
        })
    return qa_data


def generate_qa_data_from_squad(dataset: Dataset, with_context=False) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        prompt = sample["question"]
        context = sample["context"]
        answers = list(set(sample["answers"]["text"]))
        answer = answers[0] if answers else "I don't know."
        if with_context:
            prompt = context + " " + prompt
        qa_data.append({
            "context": context,
            "text": prompt + " " + answer,
            "prompt": prompt,
            "options": [],
            "answer": answer,
            "answers": answers,
        })
    return qa_data


args = read_args()
print(vars(args))
print(f"making dirs: {args.output_path}")
Path(args.output_path).mkdir(parents=True, exist_ok=True)

context = False
if args.data_name.endswith("_ctxt"):
    context = True
    data_name = args.data_name[:-5]
else:
    data_name = args.data_name
dataset_dict = load_dataset(data_name, args.subset_name)

assert isinstance(dataset_dict, dict)
for split, dataset in dataset_dict.items():
    print(f"Processing split: {split} with {len(dataset)} samples")
    if data_name == "ai2_arc":
        qa_data = generate_qa_data_from_ai2_arc(dataset)
    elif data_name == "boolq":
        qa_data = generate_qa_data_from_boolq(dataset, with_context=context)
    elif data_name == "qasc":
        qa_data = generate_qa_data_from_qasc(dataset)
    elif data_name == "squad":
        qa_data = generate_qa_data_from_squad(dataset)
    elif data_name == "squad_v2":
        qa_data = generate_qa_data_from_squad(dataset, with_context=context)
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    output_file = Path(args.output_path) / f"{split}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)
    print(f"Saved {len(qa_data)} samples to {output_file}")
