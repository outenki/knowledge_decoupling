from pathlib import Path
import sys
import json
from scipy import stats
import numpy as np


metric = sys.argv[1]
proposed_fn = Path(sys.argv[2]) / "bootstrap_analysis.json"
baseline_fn = Path(sys.argv[3]) / "bootstrap_analysis.json"
output_path = Path(sys.argv[4])
output_path.mkdir(parents=True, exist_ok=True)

print(f"Loading proposed results from {proposed_fn}")
with open(proposed_fn, "r") as f:
    proposed_results = np.array(json.load(f)[metric]["values"])
print(f"Loading baseline results from {baseline_fn}")
with open(baseline_fn, "r") as f:
    baseline_results = np.array(json.load(f)[metric]["values"])

win_rate = (proposed_results > baseline_results).mean() * 100
print(f"Win rate: {win_rate:.2f}%")
t_stat, p_value = stats.ttest_rel(proposed_results, baseline_results)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4e}")
with open(Path(sys.argv[4]) / "t_test_result.json", "w") as f:
    json.dump({
        "Win rate": win_rate,
        "T-statistic": t_stat,
        "P-value": p_value
    }, f, indent=4)
