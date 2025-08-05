"""
Evaluate model by number agreement.
Input: a dataset with sentences and a model.
Output: a dataset with sentences and the model's predictions.
"""

import torch
import argparse
import json
from pathlib import Path
import tqdm

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
from lib.utils import get_device


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', '-mp', dest='model_path', type=str, required=True,
        help='Model path to load from. (pt or safetensors)'
    )
    parser.add_argument(
        '--val-data', '-vd', dest='data_path', type=str, required=True,
        help='Path to test data. (json)'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str, required=True,
        help='Path to save results.'
    )
    return parser.parse_args()


def score_option(prompt: str, continuation: str, tokenizer, model) -> float | None:
    input_text = prompt + continuation
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]

    block_size = getattr(model.config, "n_positions", None)
    if block_size is not None and input_ids.size(1) > block_size:
        return None  # 超出最大长度限制

    input_ids = input_ids.to(model.device)
    prompt_ids = prompt_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss  # 平均交叉熵损失

    num_target_tokens = input_ids.size(1) - prompt_ids.size(1)
    total_log_prob = -loss.item() * num_target_tokens

    return total_log_prob


def score_samples(samples, tokenizer, model) -> list[dict]:
    # Score a list of samples with prompts and two options.
    filtered_samples = []
    for sample in tqdm.tqdm(samples, total=len(samples), desc="scoring samples"):
        prompt = sample["prompt"]
        option1 = sample["option1"]
        option2 = sample["option2"]

        score1 = score_option(prompt, option1, tokenizer, model)
        score2 = score_option(prompt, option2, tokenizer, model)

        if score1 is None or score2 is None:
            continue
        sample["score1"] = score1
        sample["score2"] = score2

        if score1 > score2:
            sample["pred"] = option1
        else:
            sample["pred"] = option2

        if sample["pred"] == sample["answer"]:
            sample["correct"] = True

        if sample["option1"] == sample["answer"]:
            sample["difference"] = sample["score1"] - sample["score2"]
        else:
            sample["difference"] = sample["score2"] - sample["score1"]

        filtered_samples.append(sample)
    return filtered_samples


def analyze_results(samples) -> dict:
    """
    Analyze the results of the evaluation.
    Returns a summary of the scores.
    """
    correct = 0
    total = 0
    for sample in samples:
        predicted = sample["option1"] if sample["score1"] > sample["score2"] else sample["option2"]
        is_correct = (predicted == sample["answer"])
        correct += int(is_correct)
        total += 1
    accuracy = correct / total if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def main():
    args = read_args()

    # ======== Check arguments ========
    model_path = args.model_path
    eval_data_path = args.data_path
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # ======== Set device ========
    device = get_device()
    print(f"Using device: {device}")

    # ======== Load model and tokenizer ========
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    # ======== Load evaluation data ========
    print(f"Loading evaluation data from {eval_data_path}...")
    eval_samples = []
    with open(eval_data_path, "r") as f:
        for line in tqdm.tqdm(f.readlines(), desc="Loading evaluation data"):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if not isinstance(sample, dict) or "prompt" not in sample or "option1" not in sample or "option2" not in sample:
                    raise ValueError("Each line must be a JSON object with 'prompt', 'option1', and 'option2'.")
                eval_samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    if not isinstance(eval_samples, list):
        raise ValueError("Evaluation data must be a list of samples.")

    total_count = len(eval_samples)
    print(f"Total evaluation samples: {total_count}")

    # ========= Score samples ========
    used_samples = score_samples(eval_samples, tokenizer, model)
    used_count = len(used_samples)
    print(f"Used evaluation samples (not skipped): {used_count}")
    results = analyze_results(used_samples)
    results["num_params"] = num_params

    # ========= Save results ========
    out_file = Path(out_path) / "evaluated_samples.json"
    print(f"Saving evaluation results to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(used_samples, f, indent=4)
    pd.DataFrame(used_samples).to_csv(Path(out_path) / "evaluated_samples.csv", index=False)

    out_file = Path(out_path) / "evaluation_summary.json"
    print(f"Saving summary results to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
