"""
Evaluate model by number agreement.
Input: a dataset with sentences and a model.
Output: a dataset with sentences and the model's predictions.
"""

import torch
import argparse
import json
from pathlib import Path

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', '-mp', dest='model_path', type=str,
        help='Model path to load from. (pt)'
    )
    parser.add_argument(
        '--val-data', '-vd', dest='data_name', type=str,
        help='Path to test data. (json)'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save results.'
    )
    return parser.parse_args()


def load_model(ckpt_path) -> GPT2LMHeadModel:
    """
    Load the model from the checkpoint path.
    """
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(torch.load(ckpt_path))
    return model


def score_candidate(prompt, continuation, tokenizer, model) -> float:
    # Compute log-likelihood of a candidate continuation given a prompt
    input_text = prompt + continuation
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Average loss over tokens
        total_log_prob = -loss.item() * (input_ids.size(1) - prompt_ids.size(1))  # Total log-prob of continuation
    return total_log_prob


def score_candidates(prompt, option1, option2, tokenizer, model) -> tuple[float, float]:
    """
    Score two candidates given a prompt.
    Returns the scores for both candidates.
    """
    score1 = score_candidate(prompt, option1, tokenizer, model)
    score2 = score_candidate(prompt, option2, tokenizer, model)
    return score1, score2


def score_samples(samples, tokenizer, model) -> list[dict]:
    # Score a list of samples with prompts and two options.
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        option1 = sample["option1"]
        option2 = sample["option2"]
        score1, score2 = score_candidates(prompt, option1, option2, tokenizer, model)
        samples[i]["score1"] = score1
        samples[i]["score2"] = score2
    return samples


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
    ckpt_path = args.model_path
    eval_data_path = args.data_name
    out_path = args.out_path
    if not ckpt_path or not eval_data_path or not out_path:
        raise ValueError("Please provide model path, evaluation data path, and output path.")
    if Path(ckpt_path).suffix != '.pt':
        raise ValueError("Model path must be a .pt file.")
    if Path(ckpt_path).suffix != '.json':
        raise ValueError("Model path must be a .json file.")
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True, exist_ok=True)

    # ======== Set device ========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ======== Load model and tokenizer ========
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(f"Loading model from {ckpt_path}...")
    model = load_model(ckpt_path)
    model.eval()

    # ======== Load evaluation data ========
    print(f"Loading evaluation data from {eval_data_path}...")
    with open(eval_data_path, "r") as f:
        eval_samples = json.load(f)
    if not isinstance(eval_samples, list):
        raise ValueError("Evaluation data must be a list of samples.")

    # ========= Score samples ========
    eval_samples = score_samples(eval_samples, tokenizer, model)
    results = analyze_results(eval_samples)

    # ========= Save results ========
    out_file = Path(out_path) / "evaluated_samples.json"
    print(f"Saving evaluation results to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(eval_samples, f, indent=4)

    out_file = Path(out_path) / "evaluation_summary.json"
    print(f"Saving summary results to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
