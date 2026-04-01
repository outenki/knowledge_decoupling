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
import numpy as np
from scipy import stats
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import pandas as pd
from lib.utils import get_device, print_args




def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m', dest='model', type=str, required=True,
        help='Model path to load from. (pt or safetensors)'
    )
    parser.add_argument(
        '--tokenizer', '-t', dest='tokenizer', type=str, required=False, default="gpt2"
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
        '--seed', '-sd', dest='seed', type=int, required=False, default=42,
        help='random seed.'
    )
    parser.add_argument(
        '--bootstrap', '-bs', dest='bootstrap_number', type=int, required=False, default=1,
        help='random seed.'
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
        '--mode', dest='mode', type=str, required=False, default="full", choices={"full", "simple"},
        help='The pred will be truncated if "simple" is set.'
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
    """
    1. 处理 BPE 拼接一致性
    2. 使用 Left Padding 确保对齐
    3. 矩阵化计算 CrossEntropy
    """
    # 1. 编码 Prompt 获取长度 (不带特殊 token)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)

    # 2. 拼接文本并进行 Batch Tokenize
    # 注意：某些模型如 Llama 在拼接时需要注意空格，这里统一处理
    full_texts = [prompt + " " + c for c in continuations]

    # 必须开启 Padding，且由于是 Causal LM，建议用 Left Padding
    inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    # 3. 这里的对齐逻辑是关键
    # shift_logits: 从 prompt 的最后一个 token 开始，预测 continuation 的部分
    # shift_labels: 真正的 continuation 目标 token
    # 由于存在 padding，我们需要用 mask 避开 padding 部分

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # 构建 mask：只计算 continuation 部分且不是 padding 的位置
    # 在 Left Padding 情况下，continuation 永远在最后面
    loss_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    for i in range(len(continuations)):
        # 找到当前这一行非 padding 的起始位置
        non_pad_indices = attention_mask[i].nonzero(as_tuple=True)[0]
        actual_start = non_pad_indices[0] + prompt_len - 1
        # mask 掉 prompt 部分和 padding 部分
        loss_mask[i, actual_start:] = True

    # 4. 计算 Loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    # 5. 只取 continuation 部分的平均负对数似然
    log_probs = []
    for i in range(len(continuations)):
        row_loss = token_losses[i][loss_mask[i]]
        if row_loss.numel() > 0:
            log_probs.append(-row_loss.mean().item())
        else:
            log_probs.append(-100.0)  # 容错处理

    return log_probs


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


def f1_score(pred: str, answer: str) -> tuple:
    """逐词计算 F1"""
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(answer).split()
    common = set(pred_tokens) & set(ref_tokens)
    if len(common) == 0:
        return 0, 0, 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_on_options(model, tokenizer, prompt, options, answer) -> dict:
    res = {}
    scores = [score_continuations_batch(model, tokenizer, prompt, [op])[0] for op in options]
    if any(score is None for score in scores):
        return {}
    res["scores"] = scores
    res["pred_score"] = max(scores)
    res["pred"] = options[scores.index(max(scores))]
    res["answer_score"] = scores[options.index(answer)]
    res["is_correct"] = res["pred"] == answer
    return res


def generate_answer(model, tokenizer, prompt, mode, max_new_tokens=50):
    """
    Evaluate a causal LM on a one sample.
    Automatically truncates prompt if longer than model context.
    """
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    # truncate if too long
    block_size = get_max_block_size(model)
    if block_size:
        max_input_len = block_size - max_new_tokens
    else:
        max_input_len = None
    if block_size is not None and max_input_len is not None and max_input_len is not None and input_ids.size(1) > max_input_len:
        input_ids = input_ids[:, -max_input_len:]
        attention_mask = attention_mask[:, -max_input_len:]

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    if mode == "simple":
        gen_text = gen_text.split(". ")[0]
    return gen_text


def score_on_generation(model, tokenizer, prompt, answers, mode) -> dict:
    res = {}
    pred = generate_answer(model, tokenizer, prompt, mode).lower().strip()
    res["pred"] = pred
    res["answers"] = answers
    res["pred_score"] = max(score_continuations_batch(model, tokenizer, prompt, [pred]))
    res["answer_score"] = max(score_continuations_batch(model, tokenizer, prompt, answers))
    res["is_correct"] = any([normalize_text(pred).startswith(normalize_text(answer)) for answer in answers])
    p, r, f = 0, 0, 0
    for answer in answers:
        _p, _r, _f = f1_score(pred, answer)
        if _f > f:
            p, r, f = _p, _r, _f
    res["precision"] = p
    res["recall"] = r
    res["f1"] = f
    return res


def score_samples(model, tokenizer, samples, score_on, generation_mode, few_shots="") -> list[dict]:
    # Score a list of samples with prompts and two options.
    filtered_samples = []
    # 获取模型允许的最大长度
    max_len = get_max_block_size(model) or 1024 
    # 预留给生成或选项的长度空间（比如预留 100 tokens）
    safe_threshold = max_len - 128

    for sample in tqdm.tqdm(samples, total=len(samples), desc="scoring samples"):
        prompt = few_shots + "\n" + sample["prompt"]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > safe_threshold:
            print(f" SKip: Prompt is too long: {len(prompt_ids)} > {safe_threshold}")
            continue

        answer = sample["answer"]
        answers = sample.get("answers", [answer])
        if not answers:
            answers = ["i don't know."]

        res = {}

        res = {}
        if score_on == "options":
            options = sample["options"]
            res = score_on_options(model, tokenizer, prompt, options, answer)
        elif score_on == "generation":
            res = score_on_generation(model, tokenizer, prompt, answers, generation_mode)
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
    recall = 0
    precision = 0
    for sample in samples:
        is_correct = (sample["pred"] == sample["answer"])
        correct += int(is_correct)
        total += 1
        f1 += sample.get("f1", 0)
        recall += sample.get("recall", 0)
        precision += sample.get("precision", 0)
    accuracy = correct / total if total > 0 else 0
    avg_f1 = f1 / total if total > 0 else 0
    avg_recall = recall / total if total > 0 else 0
    avg_precision = precision / total if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy, "f1": avg_f1, "precision": avg_precision, "recall": avg_recall}


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def analysis_bootstrap(results_list, confidence=0.95):
    data = np.array(results_list)
    n = len(data)

    mean = np.mean(data)
    median = np.median(data)

    std_dev = np.std(data, ddof=1)
    se = stats.sem(data)

    # Confidence Interval
    ci_t = stats.t.interval(confidence, n-1, loc=mean, scale=se)

    # 4. Bootstrap 置信区间 (更稳健，非参数方法)
    # 模拟 10,000 次重采样
    boot_means = [np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(10000)]
    ci_bootstrap = np.percentile(boot_means, [(1-confidence)/2 * 100, (1+confidence)/2 * 100])

    # p_value
    _, p_value = stats.shapiro(data)

    analysis = {
        "Count": n,
        "Mean": mean,
        "Median": median,
        "Std Dev": std_dev,
        "Std Error": se,
        f"{int(confidence*100)}% CI (T-dist)": ci_t,
        "ci_bootstrap": ci_bootstrap,
        "Shapiro-Wilk p-value": p_value,
        "Max": np.max(data),
        "Min": np.min(data)
    }

    # 打印格式化结果
    print(f"--- analysis of bootstrap (n={n}) ---")
    for key, value in analysis.items():
        if isinstance(value, (list, np.ndarray, tuple)):
            print(f"{key}: [{value[0]:.4f}, {value[1]:.4f}]")
        else:
            print(f"{key}: {value:.4f}")
    print(f"--- ----------------------------- ---")

    return {k: convert_to_serializable(v) for k, v in analysis.items()}


def main():
    args = read_args()
    print_args(vars(args))

    # ======== Check arguments ========
    model_path = args.model
    eval_data_path = args.data_path
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # ======== Set device ========
    device = get_device()
    print(f"Using device: {device}")

    # ======== Load model and tokenizer ========
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Loading model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(model_path)
    if device:
        model.to(device)

    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    # ======== Load evaluation data ========
    print(f"Loading evaluation data from {eval_data_path}...")
    eval_samples = []
    with open(eval_data_path, "r") as f:
        eval_samples = json.load(f)
        assert isinstance(eval_samples, list)

    bootstrap_acc = []
    bootstrap_f1 = []
    random.seed(args.seed)
    bootstrap_number = args.bootstrap_number
    for seed in range(bootstrap_number):
        random.seed(seed)
        out_path = out_path / f"seed_{seed}"
        out_path.mkdir(parents=True, exist_ok=True)

        if args.sample_num and len(eval_samples) > args.sample_num:
            eval_samples = random.sample(eval_samples, args.sample_num)
        total_count = len(eval_samples)
        print(f"Total evaluation samples: {total_count} with seed {seed}")

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
        used_samples = score_samples(model, tokenizer, eval_samples, args.score_on, args.mode, few_shots)
        used_count = len(used_samples)
        print(f"Evaluated on {used_count} samples")
        results = analyze_results(used_samples)
        results["num_params"] = num_params

        bootstrap_acc.append(results["accuracy"])
        bootstrap_f1.append(results["f1"])

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

        print(results)
    
    print("--- Bootstrap Analysis ---")
    bootstrap_f1_res = analysis_bootstrap(bootstrap_f1)
    bootstrap_acc_res = analysis_bootstrap(bootstrap_acc)
    out_file = Path(out_path) / "../bootstrap_analysis.json"
    print(f"Saving bootstrap analysis to {out_file}...")
    with open(out_file, "w") as f:
        json.dump({
            "f1": bootstrap_f1_res,
            "accuracy": bootstrap_acc_res
        }, f, indent=4)


if __name__ == "__main__":
    main()