import sys
from pathlib import Path

from datasets import load_from_disk

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_matched_core_ratio.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    if not Path(dataset_path).is_dir():
        print(f"Error: {dataset_path} is not a valid directory.")
        sys.exit(1)
    print(f"Loading dataset from {dataset_path}...")
    # only load the "matched_content_word_num" and "total_content_word_num" fields to save memory
    dataset = load_from_disk(dataset_path)
    dataset = dataset.select_columns([
        "token_num",
        "content_words_num",
        "replaced_ne_num",
        "replaced_unk_num"
    ])
    print(f"Loaded dataset with {len(dataset)} samples.")

    output_path = Path(dataset_path) / "matched_core_ratio.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Saved matched core ratio to {output_path}.")


if __name__ == "__main__":
    main()
