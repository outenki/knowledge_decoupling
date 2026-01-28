import argparse
import json
import re
from tqdm import tqdm
from pathlib import Path
from config import GPT_API_KEY
from openai import OpenAI

from datasets import load_dataset, Dataset

client = OpenAI(api_key=GPT_API_KEY)


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
    parser.add_argument(
        '--column-name', '-cn', dest='column_name', type=str
    )
    parser.add_argument('--format', '-f', dest='format', action='store_true')
    parser.add_argument('--format-with-options', '-op', dest='format_with_options', action='store_true')
    parser.add_argument('--probing', '-p', dest='probing', action='store_true')
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


def gpt_chat(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        timeout=10
    )
    return response.output_text.strip()
    

def convert_qa_to_probing(question: str, answer: str) -> dict:
    gpt_prompt = "把问答句转换为填空题，确保答案出现在句末。 \n"
    gpt_prompt += "input: \nWhat is the last action that the student should perform before leaving the lab area? wash hands\n"
    gpt_prompt += "output: \n"
    gpt_prompt += "<text>: The last action that the student should perform before leaving the lab area is washing hands\n"
    gpt_prompt += "<question>: The last action that the student should perform before leaving the lab area is\n"
    gpt_prompt += "<answer>: washing hands\n\n"
    gpt_prompt += "input:\n" + f"{question} {answer}\n\n"
    resp = gpt_chat(gpt_prompt).strip()
    resp = re.sub(r'\n+', '\n', resp)
    text = resp.split("<text>:")[1].split("<question>:")[0].strip()
    question = resp.split("<question>:")[1].split("<answer>:")[0].strip()
    answer = resp.split("<answer>:")[1].strip()
    return {
        "text": text,
        "question": question,
        "answer": answer
    }

def construct_data(
    context: str, question: str, options: list[str], answer: str,
    format: bool,
    format_with_options: bool,
    probing: bool
) -> dict:
    ori_context = context
    ori_question = question
    ori_options = options
    ori_answer = answer

    if probing:
        probing_text = convert_qa_to_probing(question, answer)
        text = probing_text["text"]
        question = probing_text["question"]
        answer = probing_text["answer"]
    else:
        text = question + " " + answer
    text = context + text

    
    if format:
        prompt = ""
        if context:
            prompt = f"### Context\n{context}"
        
        prompt += f"\n\n### Query\n{question}"
        if options and format_with_options:
            prompt += "\n\n### Options\n" + "\n".join(options)
        prompt += "\n\n Response\n"
    else:
        prompt = context + question

    return {
        "ori_context": ori_context,
        "ori_question": ori_question,
        "ori_options": ori_options,
        "ori_answer": ori_answer,
        "prompt": prompt,
        "answer": answer,
        "text": text
    }

def generate_qa_data_from_ai2_arc(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
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
        context = ""

        qa_data.append(
            construct_data(context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_boolq(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        if not question.endswith("?") and not question.endswith("."):
            question += "?"
        answer = "Yes." if sample["answer"] else "No."
        context = sample["passage"]
        options = ["Yes", "No"]

        qa_data.append(
            construct_data(context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_qasc(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
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
        context = ""

        
        qa_data.append(
            construct_data(context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_squad(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        context = sample["context"]
        answers = list(set(sample["answers"]["text"]))
        answer = answers[0] if answers else "I don't know."
        options = []

        qa_data.append(
            construct_data(context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


args = read_args()
print_args(vars(args))
print("")
print(f"making dirs: {args.output_path}")
Path(args.output_path).mkdir(parents=True, exist_ok=True)

print("Loading data...")
dataset_dict = load_dataset(args.data_name, args.subset_name)

assert isinstance(dataset_dict, dict)
for split, dataset in dataset_dict.items():
    if args.column_name and args.column_name != split:
        continue
    print(f"Processing sub dataset: {split} with {len(dataset)} samples")
    if args.data_name == "ai2_arc":
        qa_data = generate_qa_data_from_ai2_arc(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "boolq":
        qa_data = generate_qa_data_from_boolq(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "qasc":
        qa_data = generate_qa_data_from_qasc(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "squad_v2":
        qa_data = generate_qa_data_from_squad(
            dataset, args.format, args.format_with_options, args.probing
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}")
    output_file = Path(args.output_path) / f"{split}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)
    print(f"Saved {len(qa_data)} samples to {output_file}")
