import json
import sys
from datasets import load_dataset
from collections import defaultdict
from pathlib import Path
import tqdm

output_dir = sys.argv[1]
Path(output_dir).mkdir(parents=True, exist_ok=True)

items_per_domain = defaultdict(list)
dataset = load_dataset("ewok-core/ewok-core-1.0", split="test")

for example in dataset:
    domain = example["Domain"]
    # passed filter. add example to list
    items_per_domain[domain].append(example)
    
for domain in items_per_domain:
    with open(f"{output_dir}/{domain}.jsonl", 'w') as outfile:
        for item in tqdm.tqdm(items_per_domain[domain], desc=f"Writing {domain}"):
            outfile.write(json.dumps(item)+"\n")
            swapped_item = item
            # Separate examples where context/target is flipped. Makes it easier to compute accuracies
            swapped_item["Context1"], swapped_item["Context2"] = swapped_item["Context2"], swapped_item["Context1"]
            swapped_item["Target1"], swapped_item["Target2"] = swapped_item["Target2"], swapped_item["Target1"]
            outfile.write(json.dumps(swapped_item)+"\n")