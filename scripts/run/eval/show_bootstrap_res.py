import json
import sys
from pathlib import Path

fn = Path(sys.argv[1])/"bootstrap_analysis.json"
col_name = sys.argv[2]
with open(fn, "r") as f:
    data = json.load(f)
    if col_name == "f1":
        data = data["f1"]
    else:
        data = data["accuracy"]
    mean = data["Mean"] * 100
    ci_b, ci_t = data["ci_bootstrap"]
    std = data["Std Dev"] * 100
    print(f"{mean:.2f}" + "\interval{" + f"\pm {std:.2f}" "}")
