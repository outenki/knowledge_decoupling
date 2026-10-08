from pathlib import Path
import random
import json
from tqdm import tqdm
import argparse
import re

random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate conflict data for Google RE dataset.")
    parser.add_argument("-i", "--input-jsonl", type=str, required=True, help="Path to the input directory containing Google RE JSONL files.")
    parser.add_argument("-o", "--output-jsonl", type=str, required=True, help="Path to the output directory to save modified JSONL files.")
    return parser.parse_args()

def load_jsonl(file_path: Path) -> list[dict]:
    data = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Loading data from {file_path}"):
            data.append(json.loads(line.strip()))
    return data

def generate_conflict(sample: dict, candidates: list) -> dict:
    # Example modification: Randomly select a candidate answer that is different from the original answer
    target = sample["target_relation"]
    original_answer = sample.get("answer")
    modified_answer = random.choice([cand for cand in candidates[target] if cand != original_answer])

    # update context with new answer
    context = sample["context"]
    prompt = sample["prompt"]
    context = context.replace(original_answer, modified_answer)
    prompt = prompt.replace(original_answer, modified_answer)

    sample["answer"] = modified_answer
    sample["conflict"] = True
    sample["context"] = context
    sample["prompt"] = prompt
    return sample


def main():
    args = parse_args()
    input_file = Path(args.input_jsonl)
    output_file = Path(args.output_jsonl)
    print(f"Loading data from {input_file}")
    data = load_jsonl(input_file)

    # get candidates from answers
    candidates = {}
    for d in tqdm(data):
        target = d["target_relation"]
        answer = d["answer"]
        if target not in candidates:
            candidates[target] = [answer]
        else:
            candidates[target].append(answer)
        
    conflict_samples = []
    for sample in tqdm(data):
        conflict_samples.append(generate_conflict(sample, candidates))
    
    # save conflict samples
    print(f"Saving conflict samples to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in conflict_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()