#!/usr/bin/env python3

import argparse
import json
import statistics
from pathlib import Path
from tqdm import tqdm

from rouge_score import rouge_scorer
from bert_score import score as bert_score


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = list(f.readlines())
    for line_no, line in tqdm(enumerate(lines, 1), desc="Loading data", total=len(lines)):
        line = line.strip()
        if not line:
            continue

        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON at {path}:{line_no}: {e}"
            ) from e


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate summarization results in JSONL format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file.",
    )
    args = parser.parse_args()

    # -------------------------
    # Load data
    # -------------------------
    data = list(load_jsonl(args.input))

    print(f"Loaded {len(data)} samples.")

    if not data:
        raise ValueError("Input JSONL is empty.")

    # -------------------------
    # Extract target / response
    # -------------------------
    targets = []
    responses = []

    for i, item in enumerate(data):
        if "target" not in item:
            raise KeyError(f"Sample {i} does not contain 'target'.")

        if "resps" not in item:
            raise KeyError(f"Sample {i} does not contain 'resps'.")

        try:
            response = item["resps"][0][0]
        except (IndexError, TypeError):
            raise ValueError(
                f"Invalid 'resps' structure at sample {i}: "
                f"{item['resps']!r}"
            )
        target = item["target"]

        targets.append(target)
        responses.append(response)

    # -------------------------
    # ROUGE
    # -------------------------
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rouge_scores = []

    print("Calculating ROUGE...")

    for target, response in tqdm(zip(targets, responses), total=len(targets), desc="Scoring"):
        scores = rouge.score(target, response)

        rouge_scores.append(
            {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        )

    # -------------------------
    # BERTScore
    # -------------------------
    print("Calculating BERTScore...")

    P, R, F1 = bert_score(
        responses,
        targets,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=True,
    )

    bert_scores = []

    for i in range(len(data)):
        bert_scores.append(
            {
                "bertscore_precision": P[i].item(),
                "bertscore_recall": R[i].item(),
                "bertscore_f1": F1[i].item(),
            }
        )

    # -------------------------
    # Write output JSONL
    # -------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for target, response, rouge_score, bert_score_item in zip(
            targets,
            responses,
            rouge_scores,
            bert_scores,
        ):
            result = {}

            result["target"] = target
            result["response"] = response

            result["scores"] = {
                **rouge_score,
                **bert_score_item,
            }

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # -------------------------
    # Calculate statistics
    # -------------------------
    metrics = {
        "rouge1": [x["rouge1"] for x in rouge_scores],
        "rouge2": [x["rouge2"] for x in rouge_scores],
        "rougeL": [x["rougeL"] for x in rouge_scores],
        "bertscore_precision": [
            x["bertscore_precision"] for x in bert_scores
        ],
        "bertscore_recall": [
            x["bertscore_recall"] for x in bert_scores
        ],
        "bertscore_f1": [
            x["bertscore_f1"] for x in bert_scores
        ],
    }

    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Samples: {len(data)}")
    print()

    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 47)

    for name, values in metrics.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        print(
            f"{name:<25} "
            f"{mean:>10.4f} "
            f"{std:>10.4f}"
        )

    print("=" * 60)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()