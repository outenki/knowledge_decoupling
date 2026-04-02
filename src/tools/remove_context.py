"""
  {
    "id": "0",
    "ori_context": "Felipe de Le\u00f3n (died 1728) was a Spanish painter of the Baroque period active in Ankara .",
    "ori_question": "Where did Felipe de Le%C3%B3n die?",
    "ori_options": [],
    "ori_answer": "Ankara",
    "prompt": "Felipe de Le\u00f3n (died 1728) was a Spanish painter of the Baroque period active in Ankara .Where did Felipe de Le%C3%B3n die?",
    "answer": "Ankara",
    "text": "Felipe de Le\u00f3n (died 1728) was a Spanish painter of the Baroque period active in Ankara .Where did Felipe de Le%C3%B3n die? Ankara"
  },
"""
import sys
from pathlib import Path
import json
import tqdm


input_fn = Path(sys.argv[1])
output_fn = Path(sys.argv[2])

output_path = output_fn.parent
output_path.mkdir(parents=True, exist_ok=True)


with open(input_fn, "r") as f:
    data = json.load(f)

prepared_data = []
for item in tqdm.tqdm(data, desc="Processing data"):
    item_prompt = item["prompt"].lstrip(item["ori_context"])
    item["prompt"] = item_prompt
    item["text"] = item_prompt + " " + item["answer"]
    prepared_data.append(item)

with open(output_fn, "w") as f:
    json.dump(prepared_data, f, indent=2)