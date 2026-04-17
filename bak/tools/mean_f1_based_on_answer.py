import json
import sys


json_file = sys.argv[1]
target_answer = sys.argv[2]

with open(json_file, "r") as f:
    samples = json.load(f)

matched_f1 = []
unmatched_f1 = []
for s in samples:
    if s["answer"].lower() == target_answer.lower():
        matched_f1.append(s["f1"])
    else:
        unmatched_f1.append(s["f1"])


results = {}
results[target_answer] = (sum(matched_f1) / len(matched_f1)) if len(matched_f1) > 0 else 0
results["ohters"] = (sum(unmatched_f1) / len(unmatched_f1)) if len(unmatched_f1) > 0 else 0

print(results)


