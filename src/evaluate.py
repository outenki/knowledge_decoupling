"""
Evaluate model by number agreement.
Input: a dataset with sentences and a model.
Output: a dataset with sentences and the model's predictions.
"""

import argparse
import json
from pathlib import Path
import tqdm

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import CrossEntropyLoss

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
        '--speedup', '-su', dest='speedup', action='store_true',
        help='Enable speedup options.'
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
        return None  # skip too long samples

    input_ids = input_ids.to(model.device)
    prompt_len = prompt_ids.size(1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # p(w_{i+1}|w_{i})を予測するので、logitを[:-1]にして、labelを[:1]にする
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # mask: only calculate loss of continuation
        cont_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        cont_mask[:, prompt_len-1:] = True

        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())

        # only take the loss of continuation
        cont_losses = token_losses[cont_mask]
        avg_log_prob = -cont_losses.mean().item()  # average log-prob

    return avg_log_prob


def score_samples(samples, tokenizer, model) -> list[dict]:
    # Score a list of samples with prompts and two options.
    filtered_samples = []
    for sample in tqdm.tqdm(samples, total=len(samples), desc="scoring samples"):
        prompt = sample["prompt"]
        options = sample["options"]

        scores = [score_option(prompt, option, tokenizer, model) for option in options]
        if any(score is None for score in scores):
            continue

        sample["scores"] = scores
        sample["pred_score"] = max(scores)
        sample["pred"] = options[scores.index(max(scores))]
        sample["answer_score"] = scores[options.index(sample["answer"])]
        sample["is_correct"] = sample["pred"] == sample["answer"]

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
        is_correct = (sample["pred"] == sample["answer"])
        correct += int(is_correct)
        total += 1
    accuracy = correct / total if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def main():
    args = read_args()
    print(vars(args))

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

    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        quantization_config=None,
        device_map="auto",
    )

    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    # ======== Load evaluation data ========
    print(f"Loading evaluation data from {eval_data_path}...")
    eval_samples = []
    with open(eval_data_path, "r") as f:
        eval_samples = json.load(f)
        assert isinstance(eval_samples, list)

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
    print("accuracy:", results["accuracy"])


if __name__ == "__main__":
    main()
