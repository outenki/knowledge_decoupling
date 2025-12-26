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
from transformers import AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

import pandas as pd
from lib.utils import get_device, print_args


random.seed(42)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-mp', dest='model', type=str, required=True,
        help='Model path to load from. (pt or safetensors)'
    )
    parser.add_argument(
        '--tokenizer', '-t', dest='tokenizer', type=str, required=False,
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


def get_max_block_size(model):
    block_size = getattr(model.config, "n_positions", None)
    if not block_size:
        block_size = getattr(model.config, "max_position_embeddings", None)
    return block_size


def get_input_ids(model, tokenizer, texts):
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    # control the max length
    block_size = get_max_block_size(model)
    if block_size is not None and input_ids.size(1) > block_size:
        input_ids = input_ids[:, -block_size:]
        attention_mask = attention_mask[:, -block_size:]
    
    return input_ids.to(model.device), attention_mask.to(model.device)


def score_continuations_batch(model, tokenizer, prompt, continuations):
    texts = [prompt + " " + c for c in continuations]
    input_ids, attention_mask = get_input_ids(model, tokenizer, texts)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    max_length = get_max_block_size(model)
    log_prob = []
    for i, c in enumerate(continuations):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        ).input_ids

        cont_ids = tokenizer(
            c,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        ).input_ids

        cont_len = len(cont_ids)

        # shift
        shift_logits = logits[i, :-1, :]
        shift_labels = input_ids[i, 1:]

        # continuation mask
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


def generate_answer(model, tokenizer, prompt, max_new_tokens=50):
    """
    Evaluate a causal LM on a one sample.
    Automatically truncates prompt if longer than model context.
    """
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    # truncate if too long
    block_size = get_max_block_size(model)
    max_input_len = block_size - max_new_tokens
    if block_size is not None and input_ids.size(1) > max_input_len:
        input_ids = input_ids[:, -max_input_len:]
        attention_mask = attention_mask[:, -max_input_len:]

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return gen_text


def score_on_generation(model, tokenizer, prompt, answers) -> dict:
    res = {}
    pred = generate_answer(model, tokenizer, prompt).lower()
    res["pred"] = pred
    res["answers"] = answers
    res["pred_score"] = max(score_continuations_batch(model, tokenizer, prompt, [pred]))
    res["answer_score"] = max(score_continuations_batch(model, tokenizer, prompt, answers))
    res["is_correct"] = any([normalize_text(pred).startswith(normalize_text(answer)) for answer in answers])
    res["f1"] = max([f1_score(pred, answer) for answer in answers])
    return res


def score_samples(model, tokenizer, samples, score_on, few_shots="") -> list[dict]:
    # Score a list of samples with prompts and two options.
    filtered_samples = []
    for sample in tqdm.tqdm(samples, total=len(samples), desc="scoring samples"):
        prompt = few_shots + "\n" + sample["prompt"]
        options = sample["options"]
        answer = sample["answer"]
        answers = sample.get("answers", [answer])
        if not answers:
            answers = ["i don't know."]
        if score_on == "options":
            res = score_on_options(model, tokenizer, prompt, options, answer)
        elif score_on == "generation":
            res = score_on_generation(model, tokenizer, prompt, answers)
        sample.update(res)
        filtered_samples.append(sample)
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
    print_args(vars(args))

    # ======== Check arguments ========
    model_path = args.model
    eval_data_path = args.data_path
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # ======== Set device ========
    device = get_device()
    print(f"Using device: {device}")

    # ======== Load model and tokenizer ========
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Loading model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)

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
    used_samples = score_samples(model, tokenizer, eval_samples, args.score_on, few_shots)
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
