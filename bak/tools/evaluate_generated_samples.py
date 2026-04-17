"""
Evaluate model by number agreement.
Input: a dataset with sentences and a model.
Output: a dataset with sentences and the model's predictions.
"""

import argparse
import re
import json
from pathlib import Path
import tqdm


def print_args(args: dict):
    print("↓↓↓↓↓↓↓↓↓↓ Arguments ↓↓↓↓↓↓↓↓↓↓")
    for k, v in args.items():
        print(f"{k}: {v}")
    print("↑↑↑↑↑↑↑↑↑↑ Arguments ↑↑↑↑↑↑↑↑↑↑") 
    print()


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i', dest='input_json', type=str, required=True,
        help='Path to generated samples in json'
    )
    return parser.parse_args()


def normalize_text(s: str) -> str:
    """标准化文本：去掉标点、大小写、额外空格"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, answer: str) -> tuple[float, float, float]:
    """return precision, recall, F1"""
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(answer).split()
    common = set(pred_tokens) & set(ref_tokens)
    if len(common) == 0:
        return 0.0, 0.0, 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return precision, recall, 2 * precision * recall / (precision + recall)


def score_on_generation(sample: dict) -> dict:
    pred = sample["pred"]
    answers = sample["answers"]
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for ans in answers:
        p, r, f = f1_score(pred, ans)
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f)
    f = max(f1_scores)
    p = max(precision_scores)
    r = max(recall_scores)
    sample["f1"] = f
    sample["precision"] = p
    sample["recall"] = r
    return sample


def avg(scores: list[int]) -> float:
    sum_scores = sum(scores)
    num_scores = len(scores)
    if num_scores == 0:
        return 0
    else:
        return sum_scores / num_scores


def main():
    args = read_args()
    print_args(vars(args))

    output_path = Path(args.input_json).parent
    Path(output_path).mkdir(exist_ok=True, parents=True)

    print(f"Loading samples from {args.input_json}")
    with open(args.input_json, "r") as f:
        samples = json.load(f)

    precision_scores = []
    recall_scores = []
    f1_scores = []
    evaluated_samples = []
    for sample in tqdm.tqdm(samples, desc="Scoring"):
        evaluated = score_on_generation(sample)
        precision_scores.append(evaluated["precision"])
        recall_scores.append(evaluated["recall"])
        f1_scores.append(evaluated["f1"])
        evaluated_samples.append(evaluated)

    results = {
        "avg_precision": avg(precision_scores),
        "avg_recall": avg(recall_scores),
        "avg_f1": avg(f1_scores)
    }

    output_fn = Path(output_path) / "prf_samples.json"
    print(f"Saving evaluated samples to {output_fn}")
    with open(output_fn, "w") as f:
        json.dump(evaluated_samples, f, indent=4)

    output_fn = Path(output_path) / "avg_prf.json"
    print(f"Saving summary to {output_fn}")
    with open(output_fn, "w") as f:
        json.dump(results, f, indent=4)
    print(results)


if __name__ == "__main__":
    main()
