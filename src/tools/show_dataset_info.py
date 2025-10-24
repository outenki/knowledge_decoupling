import numpy as np
import sys
from datasets import load_from_disk

block_size = int(sys.argv[1])
dataset = load_from_disk(sys.argv[2])

def human_format(num):
    num = float(num)
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(int(num))

for dn, dt in dataset.items():
    print(f"Dataset split: {dn}")
    lines = len(dt)
    total_tokens = block_size * lines
    print(f"\tTotal samples: {human_format(lines)}")
    print(f"\tTotal tokens: {human_format(total_tokens)}")
