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