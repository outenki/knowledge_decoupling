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
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import CrossEntropyLoss

import pandas as pd
from lib.utils import get_device


random.seed(42)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', '-mp', dest='model_path', type=str, required=True,
        help='Model path to load from. (pt or safetensors)'
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
        '--score-on', '-so', dest='score_on', type=str, required=True, choices={"options", "generation"},
        help='Path to few-shots data. (json)'
    )
    parser.add_argument(
        '--example-data', '-ed', dest='example_data', type=str, required=False,
        help='Path to few-shots data. (json)'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str, required=True,
        help='Path to save results.'
    )
    return parser.parse_args()


def score_continuations_batch(model, tokenizer, prompt, continuations):
    texts = [prompt + " " + c for c in continuations]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits

    # 需要对每个样本分别计算 continuation loss
    log_prob = []
    for i, c in enumerate(continuations):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        cont_len = len(tokenizer(c, add_special_tokens=False).input_ids)

        # shift
        shift_logits = logits[i, :-1, :]
        shift_labels = input_ids[i, 1:]

        # mask continuation部分
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        mask[len(prompt_ids)-1 : len(prompt_ids)-1 + cont_len] = True

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits, shift_labels)
        cont_loss = token_losses[mask].mean()
        log_prob.append(-cont_loss.item())  # average log-prob

    return log_prob


def generate_few_shots_texts(examples: list[dict]) -> str:
    random.shuffle(examples)
    text = ""
    for _e in examples:
        _t = _e["prompt"] + ' ' + _e["answer"] + "\n"
        text += _t
    return text.strip()


def batch_generate(model, tokenizer, prompts):
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    predictions = []
    for i, prompt in enumerate(prompts):
        prompt_len = enc["input_ids"][i].size(0)
        text = tokenizer.decode(gen_ids[i][prompt_len:], skip_special_tokens=True)
        predictions.append(text)
    return predictions


def normalize_text(s: str) -> str:
    """标准化文本：去掉标点、大小写、额外空格"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, answer: str) -> float:
    """逐词计算 F1"""
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(answer).split()
    common = set(pred_tokens) & set(ref_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_on_options(model, tokenizer, prompt, options, answer) -> dict:
    res = {}
    scores = score_continuations_batch(model, tokenizer, prompt, options)
    if any(score is None for score in scores):
        return {}
    res["scores"] = scores
    res["pred_score"] = max(scores)
    res["pred"] = options[scores.index(max(scores))]
    res["answer_score"] = scores[options.index(answer)]
    res["is_correct"] = res["pred"] == answer
    return res


def score_on_generation(model, tokenizer, prompts: list[str], answers: list[list[str]]) -> dict:
    res = {}
    predictions = batch_generate(model, tokenizer, prompts)
    for _prompt, _pred, _answers in zip(prompts, predictions, answers):
        res["pred"] = _pred
        res["answers"] = _answers
        res["pred_score"] = score_continuation_batch(model, tokenizer, _prompt, [_pred])[0]
        res["answer_score"] = max(score_continuations_batch(model, tokenizer, _prompt, _answers)) if _answers else [-100]
        res["is_correct"] = any([normalize_text(pred).startswith(normalize_text(answer)) for answer in answers])
        res["f1"] = max([f1_score(pred, answer) for answer in answers])
    return res


def score_samples(model, tokenizer, samples, score_on, few_shots="") -> list[dict]:
    # Score a list of samples with prompts and two options.
    filtered_samples = []
    if score_on == "options":
        for _sample in tqdm.tqdm(samples, total=len(samples), desc="scoring samples"):
            prompt = few_shots + "\n" + _sample["prompt"]
            options = _sample["options"]
            answer = _sample["answer"]
            res = score_on_options(model, tokenizer, prompt, options, answer)

            _sample.update(res)
            filtered_samples.append(_sample)

    elif score_on == "generation":
        prompts = []
        options = []
        answers = []
        for sample in tqdm.tqdm(samples, total=len(samples), desc="scoring samples"):
            prompts.append(few_shots + "\n" + sample["prompt"]) 
            answers.append(sample["answer"])
        
        res = score_on_generation(model, tokenizer, prompts, answers)
        for _sample, _res in zip(samples, res):
            _sample.update(_res)
            filtered_samples.append(_sample)
    return filtered_samples


def analyze_results(samples) -> dict:
    """
    Analyze the results of the evaluation.
    Returns a summary of the scores.
    """
    correct = 0
    total = 0
    f1 = 0
    for sample in samples:
        is_correct = (sample["pred"] == sample["answer"])
        correct += int(is_correct)
        total += 1
        f1 += sample.get("f1", 0)
    accuracy = correct / total if total > 0 else 0
    avg_f1 = f1 / total if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy, "f1": avg_f1}


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

    if args.sample_num and len(eval_samples) > args.sample_num:
        eval_samples = random.sample(eval_samples, args.sample_num)
    total_count = len(eval_samples)
    print(f"Total evaluation samples: {total_count}")

    
    # ========= Load few shots examples ========
    few_shots = ""
    if args.example_data:
        examples = []
        print(f"Loading examples from {args.example_data}...")
        with open(args.example_data, "r") as f:
            examples = json.load(f)
        few_shots = generate_few_shots_texts(examples)
    if few_shots:
        print(f"Generated few_shots: \n{few_shots}")

    # ========= Score samples ========
    used_samples = score_samples(model, tokenizer, eval_samples, args.score_on)
    used_count = len(used_samples)
    print(f"Evaluated on {used_count} samples")
    results = analyze_results(used_samples)
    results["num_params"] = num_params

    # ========= Save results ========
    out_file = Path(out_path) / "evaluated_samples.json"
    print(f"Saving evaluated samples to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(used_samples, f, indent=4)

    out_file = Path(out_path) / "evaluated_samples.csv"
    print(f"Saving evaluated samples to {out_file}...")
    pd.DataFrame(used_samples).to_csv(out_file, index=False)

    out_file = Path(out_path) / "evaluation_summary.json"
    print(f"Saving summary to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print("accuracy:", results["accuracy"])
    print("f1:", results["f1"])


if __name__ == "__main__":
    main()
