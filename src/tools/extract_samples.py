import argparse
import json
import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input-json', '-i', dest='input_json', type=str)
parser.add_argument('--column-name', '-cn', dest='column_name', type=str)
parser.add_argument('--column-value', '-cv', dest='column_value', type=str)
parser.add_argument('--output-json', '-o', dest='output_json', type=str)
args = parser.parse_args()


output_path = Path(args.output_json).parent
output_path.mkdir(parents=True, exist_ok=True)


print(f"Loading samples from {args.input_json}")
with open(args.input_json, "r") as f:
    input_json = json.load(f)


cn = args.column_name
cv = args.column_value
output_json = [s for s in tqdm.tqdm(input_json, desc="Extracting") if s[cn] == cv]


print(f"Saving samples to {args.output_json}({len(output_json)})")
with open(args.output_json, "w") as f:
    json.dump(output_json, f, indent=4)
