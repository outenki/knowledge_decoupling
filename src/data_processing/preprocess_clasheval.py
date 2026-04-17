import sys
import json
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset

print("Loading dataset...")
ds = load_dataset("sagnikrayc/clasheval")["train"]

output_path = Path(sys.argv[1])
output_path.mkdir(parents=True, exist_ok=True)


def preprocess_sample(sample: dict) -> dict:
    # Implement your preprocessing logic here
    contexts_ori = sample["context_original"].split("\n")
    contexts_mod = sample["context_mod"].split("\n")
    ans_ori = sample["answer_original"]
    ans_mod = sample["answer_mod"]

    extracted_contexts_ori = []
    extracted_contexts_mod = []
    assert len(contexts_ori) == len(contexts_mod), "Original and modified contexts must have the same number of sentences."
    for c_ori, c_mod in zip(contexts_ori, contexts_mod):
        if c_ori != c_mod and ans_ori in c_ori and ans_mod in c_mod and c_ori.endswith(".") and c_mod.endswith("."):
            extracted_contexts_ori.append(c_ori)
            extracted_contexts_mod.append(c_mod)
    extracted_contexts_ori = " ".join(extracted_contexts_ori)
    extracted_contexts_mod = " ".join(extracted_contexts_mod)
    if not extracted_contexts_ori.strip() or not extracted_contexts_mod.strip():
        return {}
    return {
        "question": sample["question"],
        "answer_ori": ans_ori,
        "answer_mod": ans_mod,
        "context_ori": extracted_contexts_ori,
        "context_mod": extracted_contexts_mod,
    }


samples = []
for sample in tqdm(ds, desc="Preprocessing data"):
    # Process each sample
    _smpl = preprocess_sample(sample)
    if _smpl:
        samples.append(_smpl)

print(f"Preprocessed {len(samples)} samples out of {len(ds)} original samples.")
with open(output_path/"train.json", "w") as f:
    json.dump(samples, f, indent=4)
