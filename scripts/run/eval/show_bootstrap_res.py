import json
import sys
from pathlib import Path

fn = Path(sys.argv[1])/"bootstrap_analysis.json"
col_name = sys.argv[2]
with open(fn, "r") as f:
    data = json.load(f)
    if col_name == "f1":
        data = data["f1"] * 100
    else:
        data = data["accuracy"]
    mean = data["Mean"]
    ci_b, ci_t = data["ci_bootstrap"]
    ci_b = ci_b * 100
    ci_t = ci_t * 100
    print(f"{mean}({ci_b[0]:.2f}-{ci_b[1]:.2f})")
