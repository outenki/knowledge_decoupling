# - Read in a jsonl file, and read columns: "prompt", "answer", "answers", "target", "resps"
# - put "answer" and "target" to "answers" list if they are not empty
# - for each answer in "answers" list:
#   - split the answer by whitespace and comma, and check if the answer is in the "prompt" string
#   - if not all of words of the answer are in the "prompt" string, then remove the answer from the "answers" list
# - if the "answers" list is empty, then skip the sample
# - evaluate the sample with the remaining "answers" list, and write the result to a new jsonl file
#   - compute f1 score, exact match score, and partial match score for each answer in the "answers" list
#   - the evaluation is calculated by comparing the "resps" list with the "answers" list, and the best score is taken for each metric

import json
import sys
from tqdm import tqdm


def f1(prediction, ground_truth):
    prediction_tokens = prediction.replace(",", " ").split()
    ground_truth_tokens = ground_truth.replace(",", " ").split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

f_jsonl = sys.argv[1]
with open(f_jsonl, "r") as f:
    samples = [json.loads(line) for line in f]
print(f"Read {len(samples)} samples from {f_jsonl}")

# filter samples
filtered_samples = []
for sample in tqdm(samples, desc="Filtering samples"):
    doc = sample.get("doc", {})
    answers = doc.get("answers", [])
    answer = doc.get("answer")
    if answer and answer.strip() and answer not in answers:
        answers.append(answer)
    if not answers:
        target = sample.get("target")
        try:
            target = eval(target)
            if isinstance(eval(target), list):
                target = eval(target)
                answers.extend([t for t in target if isinstance(t, str) and t.strip() and t not in answers])
        except Exception as e:
            if isinstance(target, str) and target.strip() and target not in answers:
                answers.append(target)



    # filter answers that are not in the prompt
    filtered_answers = []
    for answer in answers:
        if not answer or not answer.strip() or answer.strip().lower() == "the":
            # skip empty or invalid answers
            continue
        answer_words = set(answer.replace(",", " ").split())
        prompt_words = set(doc["prompt"].replace(",", " ").split())
        if answer_words.issubset(prompt_words) or answer.strip() == "unanswerable":
            filtered_answers.append(answer)

    if len(filtered_answers) > 0:
        sample["answers"] = list(set(filtered_answers))  # Remove duplicates
        filtered_samples.append({
            "prompt": doc["prompt"],
            "answers": sample["answers"],
            "resps": sample.get("resps", []),
        })

print(f"Filtered {len(filtered_samples)} samples from {len(samples)} samples")

# evaluate samples
f1_scores = []
exact_match_scores = []
for sample in tqdm(filtered_samples, desc="Evaluating samples"):
    answers = sample["answers"]
    resps = sample.get("resps", [])
    if resps:
        resps = resps[0]
        if isinstance(resps, str):
            resps = [resps]
    best_f1 = 0
    best_exact = 0
    for answer in answers:
        for resp in resps:
            f1_score = f1(resp, answer)
            exact_match = int(resp == answer)
            if f1_score > best_f1:
                best_f1 = f1_score
            if exact_match > best_exact:
                best_exact = exact_match
    sample["best_f1"] = best_f1
    sample["best_exact"] = best_exact
    f1_scores.append(best_f1)
    exact_match_scores.append(best_exact)

print(f"Average F1 score: {sum(f1_scores) / len(f1_scores):.4f}")
print(f"Average Exact Match score: {sum(exact_match_scores) / len(exact_match_scores):.4f}")

# write to new jsonl file
f_out = f_jsonl.replace(".jsonl", "_filtered.jsonl")
print(f"Writing filtered samples to {f_out}")
with open(f_out, "w") as f:
    for sample in tqdm(filtered_samples, desc="Writing filtered samples"):
        # write json with ensure_ascii=False and indent=4
        f.write(json.dumps(sample, ensure_ascii=False, indent=4) + "\n")    