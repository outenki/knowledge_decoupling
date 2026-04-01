import numpy as np
from scipy import stats
import sys
import json
import random
from pathlib import Path
import tqdm

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

    # 4. Bootstrap ci
    boot_means = [np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(10000)]
    ci_bootstrap = np.percentile(boot_means, [(1-confidence)/2 * 100, (1+confidence)/2 * 100])

    # p_value
    _, p_value = stats.shapiro(data)

    analysis = {
        "values": data.tolist(),
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

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    bootstrap_num = int(sys.argv[1])
    target_path = Path(sys.argv[2])

    original_samples = read_json(target_path / "evaluated_samples.json")
    n_samples = len(original_samples)

    bootstrap_acc = []
    bootstrap_f1 = []

    # 2. 正确的 Bootstrap 逻辑
    for i in tqdm.tqdm(range(bootstrap_num), desc="Bootstrapping"):
        random.seed(i)
        resampled_data = random.choices(original_samples, k=n_samples)

        # 计算这一批次的指标
        f1_i = [r.get("f1", 0) for r in resampled_data]
        acc_i = [1 if r.get("is_correct", False) else 0 for r in resampled_data]

        bootstrap_f1.append(np.mean(f1_i))
        bootstrap_acc.append(np.mean(acc_i))

    print("--- Bootstrap Analysis on f1---")
    bootstrap_f1_res = analysis_bootstrap(bootstrap_f1)
    print("--- Bootstrap Analysis on accuracy---")
    bootstrap_acc_res = analysis_bootstrap(bootstrap_acc)

    out_file = Path(target_path) / "bootstrap_analysis.json"
    print(f"Saving bootstrap analysis to {out_file}...")
    with open(out_file, "w") as f:
        json.dump({
            "f1": bootstrap_f1_res,
            "accuracy": bootstrap_acc_res
        }, f, indent=4)
