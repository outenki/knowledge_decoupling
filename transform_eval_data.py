import json
import sys

def convert_example(example: dict) -> dict:
    """
    把带有 option1, option2, ... 的样本转换为带有 options 列表的样本
    """
    # 收集所有以 "option" 开头的字段
    options = []
    i = 1
    while f"option{i}" in example:
        options.append(example[f"option{i}"])
        i += 1

    # 构造新格式
    return {
        "text": example["text"],
        "prompt": example["prompt"],
        "options": options,
        "answer": example["answer"],
    }


# read data
eval_data_path = sys.argv[1]  # e.g., input/evaluate_data/qa_arc_easy/test.jsonl
out_path = sys.argv[2]  # e.g., input/evaluate_data/qa_arc_easy/test_converted.json

eval_samples = []
with open(eval_data_path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
            assert isinstance(sample, dict)
            assert "prompt" in sample and "option1" in sample and "option2" in sample
            converted_sample = convert_example(sample)
            eval_samples.append(converted_sample)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except AssertionError as e:
            print(f"Invalid sample format: {e}")

with open(out_path, "w") as f:
    json.dump(eval_samples, f, indent=2)
print(f"Converted {len(eval_samples)} samples and saved to {out_path}")