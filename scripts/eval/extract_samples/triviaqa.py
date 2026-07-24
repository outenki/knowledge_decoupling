import json
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm


in_json = sys.argv[1]

output_path = Path(in_json).parent
output_stem = Path(in_json).stem
in_name = Path(in_json).name

samples = []
with open(in_json, "r") as f:
    for line in tqdm(f, desc=f"Loading data from {in_name}", total=sum(1 for _ in open(in_json, 'r'))):
      samples.append(json.loads(line))

extracted = []
for spl in tqdm(samples, total=len(samples), desc="Extracting samples"):
    extracted.append({
        "doc_id": spl["doc_id"],
        "prompt": spl["arguments"]["gen_args_0"]["arg_0"],
        "response": spl["resps"][0][0],
        "filtered_resps": spl["filtered_resps"][0],
        "target": spl["target"]
    })


# save json samples
print("Saving json samples...")
output_fn = output_path/f"simpled_{output_stem}.json"
with open(output_fn, "w") as f:
  json.dump(extracted, f, indent=4)


# save csv samples
print("Saving csv samples...")
output_fn = output_path/f"simpled_{output_stem}.csv"
df = pd.DataFrame(extracted)
df.to_csv(output_fn, index=False)
