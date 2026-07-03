import argparse

from datasets import DatasetDict
from transformers import AutoTokenizer
import tqdm

from src.lib.dataset import load_custom_dataset, select_data_by_indices


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', dest='data', type=str, help='Dataset path to load from.')
    parser.add_argument('--column', '-c', dest='column', type=str, help='Column name to count.')
    parser.add_argument('--split', '-sp', dest='split', type=str, default="train", help='Dataset split name to process.') 
    parser.add_argument('--load-from', '-lf', dest='load_from', choices=["hf", "local"])
    return parser.parse_args()


args = read_args()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# ========  Load dataset ========
print("**** Loading dataset...")
dataset = load_custom_dataset(
    data_name=args.data,
    data_type=None,
    load_from=args.load_from
)
print(f"Dataset loaded with {dataset.num_rows} samples.")
if isinstance(dataset, DatasetDict):
    dataset = dataset[args.split]


total = 0

for example in tqdm.tqdm(dataset, total=len(dataset), desc="Counting tokens"):
    total += len(
        tokenizer(example[args.column], add_special_tokens=False,)["input_ids"]
    )

print(total)