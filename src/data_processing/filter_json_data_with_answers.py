# only extract samples with answers
import json
from pathlib import Path
import sys
import tqdm

def filter_json_data_with_answers(input_path: Path, output_path: Path):
    if input_path.is_dir():
        for json_file in input_path.rglob("*.json"):
            print(f"Filtering {json_file}...")
            with open(json_file, "r") as f:
                data = json.load(f)
            filtered_data = [item for item in tqdm.tqdm(data) if item.get("answer") and item.get("answer").strip().lower().strip(".") != "i don't know"]
            output_file = output_path / json_file.name
            with open(output_file, "w") as f:
                json.dump(filtered_data, f, indent=2)
    else:
        print(f"Filtering {input_path}...")
        with open(input_path, "r") as f:
            data = json.load(f)
        filtered_data = [item for item in tqdm.tqdm(data) if item.get("answer") and item.get("answer").strip().lower().strip(".") != "i don't know"]
        with open(output_path, "w") as f:
            json.dump(filtered_data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_json_data_with_answers.py <input_path> <output_path>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    output_path.mkdir(exist_ok=True, parents=True)
    filter_json_data_with_answers(input_path, output_path)