import sys
import json
import csv
from pathlib import Path


jsonl_path = Path(sys.argv[1])
csv_path = jsonl_path.with_suffix(".csv")

with jsonl_path.open("r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

fieldnames = data[0].keys()

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"Saved to: {csv_path}")