import argparse
import json
from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset, Dataset


def print_args(args: dict):
    print("↓↓↓↓↓↓↓↓↓↓ Arguments ↓↓↓↓↓↓↓↓↓↓")
    for k, v in args.items():
        print(f"{k}: {v}")
    print("↑↑↑↑↑↑↑↑↑↑ Arguments ↑↑↑↑↑↑↑↑↑↑") 


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str, required=True,
        choices=['ai2_arc', 'boolq', 'qasc', "squad_v2"],
        help='Name of the dataset to load from Hugging Face'
    )
    parser.add_argument('--with-options', '-op', dest='with_options', action='store_true')
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


def generate_qa_data_from_ai2_arc(dataset: Dataset, with_options: bool) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        if not question.endswith("?") and not question.endswith("."):
            question += "?"
        
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        options = sample["choices"]["text"]
        answer = options[labels.index(answer_key)]
        text = question + "\n" + answer

        prompt = f"### Query\n{question}"
        if with_options:
            prompt += "\n\n### Options\n" + "\n".join(sample["choices"]["text"])
        prompt += "\n\n### Response\n"
        qa_data.append({
            "context": "",
            "text": text,
            "options": options,
            "prompt": prompt,
            "answer": answer
        })
    return qa_data


def generate_qa_data_from_boolq(dataset: Dataset, with_options) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        if not question.endswith("?") and not question.endswith("."):
            question += "?"
        answer = "Yes." if sample["answer"] else "No."
        text = question + "\n" + answer

        context = sample["passage"]
        text = context + "\n" + text 
        prompt = f"### Context\n{context}\n\n"
        prompt += f"### Query\n{question}"
        if with_options:
            prompt += f"### Options\n" + "\n".join(["Yes.", "No."])
        prompt += "\n\n### Response\n"
        qa_data.append({
            "context": sample["passage"],
            "options": ["Yes.", "No."],
            "text": text,
            "prompt": prompt,
            "answer": answer
        })
    return qa_data


def generate_qa_data_from_qasc(dataset: Dataset, with_options: bool) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        if not question.endswith("?") and not question.endswith("."):
            question += "?"
        options = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        if answer_key not in labels:
            continue  # Skip if the answer key is not in the labels
        answer = options[labels.index(answer_key)]
        text = question + "\n" + answer
        
        prompt = f"### Query\n{question}"
        if with_options:
            prompt += f"### Options\n" + "\n".join(options)
        prompt += "\n\n### Response\n"

        qa_data.append({
            "context": "",
            "text": text,
            "prompt": prompt,
            "options": options,
            "answer": answer
        })
    return qa_data


def generate_qa_data_from_squad(dataset: Dataset) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        context = sample["context"]
        answers = list(set(sample["answers"]["text"]))
        answer = answers[0] if answers else "I don't know."
        text = question + "\n" + answer

        text = context + "\n" + text
        prompt = f"### Context\n{context}\n\n"
        prompt += f"### Query\n{question}"
        prompt += f"\n\n### Response\n"
        qa_data.append({
            "context": context,
            "text": text,
            "prompt": prompt,
            "options": [],
            "answer": answer,
            "answers": answers,
        })
    return qa_data


args = read_args()
print_args(vars(args))
print("")
print(f"making dirs: {args.output_path}")
Path(args.output_path).mkdir(parents=True, exist_ok=True)

# context = False
# if args.data_name.endswith("_ctxt"):
#     context = True
#     data_name = args.data_name[:-5]
# else:
#     data_name = args.data_name
print("Loading data...")
dataset_dict = load_dataset(args.data_name, args.subset_name)

assert isinstance(dataset_dict, dict)
for split, dataset in dataset_dict.items():
    print(f"Processing split: {split} with {len(dataset)} samples")
    if args.data_name == "ai2_arc":
        qa_data = generate_qa_data_from_ai2_arc(dataset, with_options=args.with_options)
    elif args.data_name == "boolq":
        qa_data = generate_qa_data_from_boolq(dataset, with_options=args.with_options)
    elif args.data_name == "qasc":
        qa_data = generate_qa_data_from_qasc(dataset, with_options=args.with_options)
    # elif args.data_name == "squad":
    #     qa_data = generate_qa_data_from_squad(dataset)
    elif args.data_name == "squad_v2":
        qa_data = generate_qa_data_from_squad(dataset)
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}")
    output_file = Path(args.output_path) / f"{split}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)
    print(f"Saved {len(qa_data)} samples to {output_file}")
