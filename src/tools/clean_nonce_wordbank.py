import argparse
import json


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i', dest='input', type=str,
        help='input wordbank json file'
    )
    parser.add_argument(
        '--output', '-o', dest='output', type=str,
        help='output wordbank json file'
    )
    args = parser.parse_args()
    return args


def clean(values) -> list:
    cleaned = set()
    for word, lemma in values:
        cleaned.add((word.lower(), lemma))
    return [[w, l] for w, l in cleaned]


def main():
    args = read_args()
    with open(args.input, 'r') as f:
        wb = json.load(f)

    wb = {k: clean(v) for k, v in wb.items()}
    with open(args.output, 'w') as f:
        json.dump(wb, f, indent=4)
