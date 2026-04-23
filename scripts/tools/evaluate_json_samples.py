"""
    {
    "text": "What process makes it possible for the nutrients from organic material to be added to soil? decomposition",
    "prompt": "What process makes it possible for the nutrients from organic material to be added to soil?",
    "options": [
        "weathering",
        "erosion",
        "decomposition",
        "succession"
    ],
    "answer": "decomposition",
    "pred": "decomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition\ndecomposition",
    "answers": [
        "decomposition"
    ],
    "pred_score": -1.546875,
    "answer_score": -0.000614166259765625,
    "is_correct": true,
    "precision": 0.07692307692307693,
    "recall": 1.0,
    "f1": 0.14285714285714288
},
"""

import json
import sys
import re


def normalize_text(s: str) -> str:
    """标准化文本：去掉标点、大小写、额外空格"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, answer: str) -> tuple:
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(answer).split()
    common = set(pred_tokens) & set(ref_tokens)
    if len(common) == 0:
        return 0, 0, 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def calculate_f1(sample: dict) -> float:
    answers = sample.get("answers", [])
    pred = sample.get("pred", "")
    pred = pred.split("\n")[0]
    f1_scores = []
    for answer in answers:
        _, _, f1 = f1_score(pred, answers[0]) if answers else (0, 0, 0)
        f1_scores.append(f1)
    return max(f1_scores)
    

fn = sys.argv[1]

with open(fn, 'r') as f:
    data = json.load(f)


f1_scores = [calculate_f1(sample) for sample in data]
avg_f1 = sum(f1_scores) / len(f1_scores)
print(f"Average F1 Score: {avg_f1:.4f}")