import sys
from pathlib import Path
import multiprocessing

from datasets import load_from_disk


CPU_NUM: int = min(8, multiprocessing.cpu_count())

def filter(example) -> bool:
    ent_num = example["replaced_ne_num"] 
    unk_num = example["replaced_unk_num"] 
    token_num = example["token_num"] 
    replace_ratio = (ent_num + unk_num) / token_num
    return replace_ratio <= 0.05


def main():
    if len(sys.argv) != 3:
        print("Usage: python filter_dataset_by_replace_ratio.py <dataset_path> <output_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    Path(output_path).mkdir(parents=True, exist_ok=True)


    if not Path(dataset_path).is_dir():
        print(f"Error: {dataset_path} is not a valid directory.")
        sys.exit(1)
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    num_before = len(dataset)
    print(f"Loaded dataset with {len(dataset)} samples.")


    filtered = dataset.filter(filter, num_proc=CPU_NUM, desc="Filtering dataset by replace ratio")
    num_after = len(filtered)

    print(f"Samples after filtering:  {num_after} ({num_after / num_before * 100:.2f}%)")

    print(f"Saving dataset to {output_path}")
    filtered.save_to_disk(output_path)


if __name__ == "__main__":
    main()
