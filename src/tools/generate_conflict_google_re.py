from pathlib import Path
import random
import json
from tqdm import tqdm


random.seed(42)

def load_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Loading data from {file_path}", total=sum(1 for _ in open(file_path, 'r'))):
            data.append(json.loads(line.strip()))
    return data


def load_google_re(fn) -> list[dict]:
    data = []
    _data = load_jsonl(fn)
    data.extend(_data)
    return data


def extract_candidate_labels(data: list[dict]) -> list[dict]:
    candidate_labels = list()
    for sample in data:
        candidate_labels.append({"obj_label": sample["obj_label"], "obj_aliases": sample["obj_aliases"]})
    return candidate_labels


def replace_obj_label(data: list[dict], candidate_labels: list[dict]) -> list[dict]:
    for sample in data:
        # Randomly select a candidate label
        while True:
            candidate = random.choice(candidate_labels)
            # Replace the obj_label and obj_aliases with the candidate's
            if candidate["obj_label"] != sample["obj_label"] and not set(candidate["obj_aliases"]).intersection(set(sample["obj_aliases"])):
                sample["ori_obj_label"] = sample["obj_label"]
                sample["ori_obj_aliases"] = sample["obj_aliases"]
                sample["mod_obj_label"] = candidate["obj_label"]
                sample["mod_obj_aliases"] = candidate["obj_aliases"]
                break
    return data


if __name__ == "__main__":
    input_dir = Path("/home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE/")
    output_dir = Path("/home/pj25000107/ku50001566/projects/knowledge_decoupling/data/Google_RE_conflict/")
    for fn in ["place_of_birth_test.jsonl", "date_of_birth_test.jsonl", "place_of_death_test.jsonl"]:
        print("Processing file:", fn)
        input_fn = input_dir / f"{fn}"
        output_fn = output_dir / f"{fn}"
        data = load_google_re(input_fn)
        candidate_labels = extract_candidate_labels(data)
        modified_data = replace_obj_label(data, candidate_labels)

        with open(output_fn, 'w') as f:
            for sample in modified_data:
                f.write(json.dumps(sample) + '\n')
