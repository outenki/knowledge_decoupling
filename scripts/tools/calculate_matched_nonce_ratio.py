# Read a transformer dataset
# sample 100000 samples randomly
# Read the "matched_content_word_num" and "total_content_word_num" fields
# Calculate the ratio of "matched_content_word_num" / "total_content_word_num"
# Print the average ratio

import sys
import random
from pathlib import Path
from tqdm import tqdm

from datasets import load_from_disk

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_matched_nonce_ratio.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    if not Path(dataset_path).is_dir():
        print(f"Error: {dataset_path} is not a valid directory.")
        sys.exit(1)
    print(f"Loading dataset from {dataset_path}...")
    # only load the "matched_content_word_num" and "total_content_word_num" fields to save memory
    dataset = load_from_disk(dataset_path)
    dataset = dataset.select_columns(["matched_content_word_num", "total_content_word_num"])
    print(f"Loaded dataset with {len(dataset)} samples.")

    # randomly sample 100000 samples
    sample_size = min(100000, len(dataset))
    print(f"Sampling {sample_size} samples from the dataset...")
    indices = random.sample(range(len(dataset)), sample_size)
    dataset = dataset.select(indices)
    print(f"Sampled {len(dataset)} samples.")

    print("Calculating matched nonce ratios...")
    matched_content_word_num = dataset['matched_content_word_num']
    total_content_word_num = dataset['total_content_word_num']

    ratios = [m / t if t > 0 else 0 for m, t in tqdm(zip(matched_content_word_num, total_content_word_num), total=len(dataset))]

    # print the average ratio
    average_ratio = sum(ratios) / len(ratios)
    print(f"Average matched nonce ratio: {average_ratio:.4f}")

    for target_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
        count_high_ratio = sum(1 for r in tqdm(ratios, desc=f"Checking ratios < {target_ratio}") if r < target_ratio)
        print(f"Number of samples with matched nonce ratio < {target_ratio}: {count_high_ratio} / {len(ratios)}")

if __name__ == "__main__":
    main()
