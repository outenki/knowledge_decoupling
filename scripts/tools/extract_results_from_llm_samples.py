import json
import sys
from collections import Counter

with open(sys.argv[1], "r") as f:
    samples = [json.loads(l) for l in f]

counter = Counter()

for sample in samples:
    scores = [float(x[0]) for x in sample["filtered_resps"]]
    pred = max(range(len(scores)), key=lambda i: scores[i])
    counter[pred] += 1

print(counter)