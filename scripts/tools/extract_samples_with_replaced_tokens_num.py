import argparse
import json
import pdb
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract samples with replaced tokens and count the number of replaced tokens.")
    parser.add_argument("-s1", "--sample1", type=str, required=True, help="Path to the sample1 JSONL file.")
    parser.add_argument("-s2", "--sample2", type=str, required=True, help="Path to the sample2 JSONL file.")
    parser.add_argument("-rp", "--replaced-jsonl", type=str, required=True, help="Path to the replaced tokens JSONL file.")
    parser.add_argument("-m", "--metric", type=str, required=True, choices={"acc", "f1", "em"}, help="Metric to use for evaluation.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output csv file.")
    return parser.parse_args()



def get_acc(sample: dict):
    return sample["acc"]


def calculate_f1(target: str, pred: str):
    target_tokens = target.strip().split()
    pred_tokens = pred.strip().split()

    common = set(target_tokens) & set(pred_tokens)
    num_common = len(common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(target_tokens)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def get_f1(sample: dict):
    target = sample["target"]
    prediction = sample["f1"][0]["prediction_text"]
    return calculate_f1(target, prediction)


def get_em(sample: dict):
    return sample["exact_match"]


def valid_samples(doc) -> bool:
    descriptions = [
        d.strip()
        for d in doc["search_results"]["description"][:5]
        if d and d.strip()
    ]

    contexts = [
        c.strip()
        for c in doc["search_results"]["search_context"][:5]
        if c and c.strip()
    ]
    return len(descriptions) > 0 or len(contexts) > 0

def extract_sample(sample:dict, metric: str) -> dict:
    res = {
        "doc_id": sample["doc_id"],
        "prompt": sample["arguments"]["gen_args_0"]["arg_0"].replace("??", "?").replace("\n\n", "\n").strip(),
        # "prompt": sample["doc"].get("sentence", ""),
        "answerable": sample["target"].strip() != "unanswerable"
    }
    if metric == "acc":
        res[metric] = get_acc(sample)
    elif metric == "f1":
        res[metric] = get_f1(sample)
    elif metric == "em":
        res[metric] = get_em(sample)
    return res


def extract_replaced_tokens_count(replaced_tokens_sample: dict) -> dict:
    res = {
        "doc_id": replaced_tokens_sample["id"],
        "prompt": replaced_tokens_sample["prompt_ori"],
        "replaced_tokens_count": replaced_tokens_sample["replaced_total_num"],
        "ent_tokens_count": replaced_tokens_sample["replaced_ent_num"],
        "replaced_ratio": 0,
        "ent_ratio": 0,
        "content_words_num": replaced_tokens_sample["content_words_num"],
    }
    res["replaced_ratio"] = res["replaced_tokens_count"] / res["content_words_num"] if res["content_words_num"] > 0 else 0
    res["ent_ratio"] = res["ent_tokens_count"] / res["content_words_num"] if res["content_words_num"] > 0 else 0
    return res


def load_jsonl(file_path: str) -> list:
    print(f"Loading JSONL data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = list(f.readlines())
    return [json.loads(line) for line in tqdm(lines, desc="Loading JSONL", unit="lines")]


def main():
    args = parse_args()

    sample1_data = load_jsonl(args.sample1)
    sample2_data = load_jsonl(args.sample2)
    replaced_tokens_data = load_jsonl(args.replaced_jsonl)
    if args.metric == "em":
        replaced_tokens_data = [sample for sample in replaced_tokens_data if sample["content_words_num"] > 0]
        # reset doc id to string for replaced_tokens_data
        for i in range(len(replaced_tokens_data)):
            replaced_tokens_data[i]["id"] = str(i)

    # Create a dictionary for quick lookup of samples by doc_id
    sample1_dict = {sample["arguments"]["gen_args_0"]["arg_0"].replace("??", "?"): sample for sample in tqdm(sample1_data, desc="Creating sample1 dict")}
    sample2_dict = {sample["arguments"]["gen_args_0"]["arg_0"].replace("??", "?"): sample for sample in tqdm(sample2_data, desc="Creating sample2 dict")}
    # sample1_dict = {sample["doc"]["sentence"]: sample for sample in tqdm(sample1_data, desc="Creating sample1 dict")}
    # sample2_dict = {sample["doc"]["sentence"]: sample for sample in tqdm(sample2_data, desc="Creating sample2 dict")}
    replaced_tokens_dict = {sample["prompt_ori"]: sample for sample in tqdm(replaced_tokens_data, desc="Creating replaced tokens dict")}

    # Extract relevant information and calculate metrics
    extracted_samples = []
    for prompt, replaced_sample in tqdm(replaced_tokens_dict.items(), desc="Processing replaced tokens samples", total=len(replaced_tokens_dict)):
        # prompt = prompt.split(":")[1].split("\n")[0].strip() 
        prompt = prompt.replace("??", "?").replace("\n\n", "\n").strip()  # Normalize prompt by replacing double question marks and extra spaces
        # import ipdb; ipdb.set_trace()
        if prompt in sample1_dict and prompt in sample2_dict:
            sample1 = extract_sample(sample1_dict[prompt], args.metric)
            sample2 = extract_sample(sample2_dict[prompt], args.metric)
            replaced_info = extract_replaced_tokens_count(replaced_sample)
            # assert on prompt to ensure they match
            assert sample1["prompt"] == sample2["prompt"] == prompt, f"Prompt mismatch for doc_id {prompt}"

            combined_sample = {
                # "doc_id": doc_id,
                "prompt": replaced_info["prompt"],
                "replaced_tokens_count": replaced_info["replaced_tokens_count"],
                "content_words_num": replaced_info["content_words_num"],
                "ent_tatio": replaced_info["ent_ratio"],
                "replaced_ratio": replaced_info["replaced_ratio"],
                "ent_tatio_rounded": round(replaced_info["ent_ratio"], 1),
                "replaced_ratio_rounded": round(replaced_info["replaced_ratio"], 1),
                f"sample1_{args.metric}": sample1[args.metric],
                f"sample2_{args.metric}": sample2[args.metric],
            }
            extracted_samples.append(combined_sample)

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(extracted_samples)
    df.to_csv(args.output, index=False)
    print(f"Extracted samples saved to {args.output}")
    print()


if __name__ == "__main__":
    main()
