"""
  {
    "text": "Students study human body structure to learn how the body functions. Which life-size model would best represent the size, shape, and location of human internal organs? a three-dimensional plastic upper body with removable parts",
    "prompt": "Students study human body structure to learn how the body functions. Which life-size model would best represent the size, shape, and location of human internal organs?",
    "options": [
      "a two-dimensional upper body diagram with magnetic stickers of the organs",
      "a three-dimensional plastic upper body with removable parts",
      "a two-dimensional detailed wall poster",
      "a three-dimensional paper body"
    ],
    "answer": "a three-dimensional plastic upper body with removable parts",
    "id": 342
  },
"""
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
sep_token = sys.argv[3]


output_path.mkdir(exist_ok=True, parents=True)


def sep_prompt_response(example, sep_token):
    prompt = example["prompt"]
    prompt = prompt + " " + sep_token
    prompt = prompt.strip() + " "
    example["prompt"] = prompt
    return example

for split_name in ["train", "dev", "test"]:
    split_path = input_path / f"{split_name}.json"
    print(f"loading data from {split_path}")
    if not split_path.exists():
        print(f"file {split_path} does not exist, skipping")
        continue
    with open(split_path, "r") as f:
        data = json.load(f)

    separated_data = [sep_prompt_response(example, sep_token) for example in data]

    print(f"saving separated data to {output_path / f'{split_name}.json'}")
    with open(output_path / f"{split_name}.json", "w") as f:
        json.dump(separated_data, f)
