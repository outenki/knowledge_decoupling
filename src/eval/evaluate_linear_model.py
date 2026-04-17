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
import random

import torch
from transformers import AutoTokenizer
from transformers import AutoModel

import pandas as pd
from lib.utils import get_device, print_args
from lib.linear_model import MCQModel


random.seed(42)


def read_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--init-model-path', '-im', dest='init_model_path', type=str, required=True,
    #     help='Initial model path to load from. (pt or safetensors)'
    # )
    parser.add_argument(
        '--model-path', '-p', dest='model_path', type=str, required=True,
        help='Linear weights path to load from. (pt or safetensors)'
    )
    parser.add_argument(
        '--test-data', '-vd', dest='data_path', type=str, required=True,
        help='Path to test data. (json)'
    )
    parser.add_argument(
        '--sample-num', '-sn', dest='sample_num', type=int, required=False,
        help='Max number of samples to use.'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str, required=True,
        help='Path to save results.'
    )
    return parser.parse_args()


def encode_mcq(question, options, tokenizer):
    texts = [question + " " + opt for opt in options]
    enc = [tokenizer(t)["input_ids"] for t in texts]
    max_len = max(len(x) for x in enc)
    pad_id = tokenizer.pad_token_id
    enc = [x + [pad_id] * (max_len - len(x)) for x in enc]
    return enc

def predict(model, input_ids, device):
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids)["logits"]

    pred = torch.argmax(logits, dim=-1).item()
    scores = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    return pred, scores


def main():
    args = read_args()
    print_args(vars(args))

    # ======== Check arguments ========
    eval_data_path = args.data_path
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # ======== Set device ========
    device = get_device()
    print(f"Using device: {device}")

    # ======== Load model and tokenizer ========
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded from {args.model_path}")
    init_model = AutoModel.from_pretrained(args.model_path)
    checkpoint = torch.load(f"{args.model_path}/model.pt", map_location="cpu")
    model = MCQModel(init_model, num_choices=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {args.model_path}")

    model.eval()
    model.to(device)

    # ======== Load evaluation data ========
    print(f"Loading evaluation data from {eval_data_path}...")
    eval_samples = []
    with open(eval_data_path, "r") as f:
        eval_samples = json.load(f)
        assert isinstance(eval_samples, list)
    
    option_num = len(eval_samples[0]["options"])
    print(f"Number of options per question: {option_num}")
    eval_samples = [s for s in eval_samples if len(s["options"]) == option_num]

    if args.sample_num and len(eval_samples) > args.sample_num:
        eval_samples = random.sample(eval_samples, args.sample_num)
    total_count = len(eval_samples)
    print(f"Total evaluation samples: {total_count}")

    # ======== Evaluate ========
    correct = 0
    total = 0

    evaluated_samples = []
    for sample in eval_samples:
        input_ids = encode_mcq(sample["prompt"], sample["options"], tokenizer)
        pred, scores = predict(model, input_ids, device)
        sample["pred"] = pred
        sample["scores"] = scores
        evaluated_samples.append(sample)
        if sample["options"][pred] == sample["answer"]:
            correct += 1
        total += 1

    print("Accuracy:", correct / total)

    # ========= Save results ========
    out_file = Path(out_path) / "evaluated_samples.json"
    print(f"Saving evaluated samples to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(evaluated_samples, f, indent=4)

    results = {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total
    }
    out_file = Path(out_path) / "evaluation_summary.json"
    print(f"Saving summary to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(results)


if __name__ == "__main__":
    main()