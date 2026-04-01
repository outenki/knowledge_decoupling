from pathlib import Path
import sys
import json
from scipy import stats


metric = sys.argv[1]
proposed_fn = Path(sys.argv[2]) / "bootstrap_analysis.json"
baseline_fn = Path(sys.argv[3]) / "bootstrap_analysis.json"
output_path = Path(sys.argv[4])

with open(proposed_fn, "r") as f:
    proposed_results = json.load(f)[metric]["values"]
with open(baseline_fn, "r") as f:
    baseline_results = json.load(f)[metric]["values"]

# 执行配对 t 检验
t_stat, p_value = stats.ttest_rel(proposed_results, baseline_results)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4e}")
with open(Path(sys.argv[4]) / "t_test_result.json", "w") as f:
    json.dump({
        "T-statistic": t_stat,
        "P-value": p_value
    }, f, indent=4)
