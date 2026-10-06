#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from datasets import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--column", required=True, help="Column to extract")
    parser.add_argument("--output", required=True, help="Output Hugging Face Dataset directory")
    args = parser.parse_args()

    values = []

    with open(args.input, "r", encoding="utf-8") as f:
        lines = list(f.readlines())

    for line_num, line in enumerate(tqdm(lines, desc="Processing JSONL"), 1):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

        if args.column not in data:
            raise KeyError(
                f"Column '{args.column}' not found at line {line_num}"
            )

        values.append(data[args.column])

    dataset = Dataset.from_dict({
        "text": values
    })

    Path(args.output).mkdir(exist_ok=True, parents=True)
    dataset.select(range(min(50, len(dataset)))).to_json(
        Path(args.output) / "example_sentences.json"
    )
    dataset.save_to_disk(args.output)

    print(f"Saved {len(dataset)} examples to {args.output}")
    print(dataset)


if __name__ == "__main__":
    main()