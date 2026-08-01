# conver '<ABC_ID_OOOO>' to `ABC_ID_OOOO`
import re
import sys
from pathlib import Path
import json
from tqdm import tqdm


def _clean_angle(text: str|list) -> str|list:
    if isinstance(text, str):
        return re.sub(
            r'<([A-Za-z_]+_\d+_\d+)>',
            r'\1',
            text,
        )
    if isinstance(text, list):
        return [
            re.sub(
                r'<([A-Za-z_]+_\d+_\d+)>',
                r'\1',
                t,
            ) for t in text
        ]


input_json_fn = sys.argv[1]
output_json_fn = sys.argv[2]

if not Path(input_json_fn).exists():
    print(f"{input_json_fn} does not exist. Exit.")
    exit()

Path(output_json_fn).parent.mkdir(parents=True, exist_ok=True)


# Load json
print(f"Loading json from {input_json_fn}")
with open(input_json_fn, "r") as f:
    input_json = json.load(f)

output_json = []
for item in tqdm(input_json, desc="Removing angles"):
    for k,v in item.items():
        item[k] = _clean_angle(v)
    output_json.append(item)

print(f"Saving json to {output_json_fn}")
with open(output_json_fn, "w") as f:
    json.dump(output_json, f, indent=4)