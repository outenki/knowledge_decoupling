import argparse
import json
import os
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="Path to input JSONL file")
    parser.add_argument("--train-ratio", type=float, help="Ratio of training data, e.g. 0.9")
    parser.add_argument("--output-dir", type=str, help="Directory to save train.jsonl and val.jsonl")

    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # 固定随机种子，保证结果可复现
    random.seed(42)

    # --------------------------------------------------
    # 1. 读取数据
    # --------------------------------------------------
    data = []

    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data.append(json.loads(line))

    print(f"Total samples: {len(data)}")

    # --------------------------------------------------
    # 2. 按 context 分组
    # --------------------------------------------------
    context_to_samples = defaultdict(list)

    for item in data:
        context = item["context"]
        context_to_samples[context].append(item)

    contexts = list(context_to_samples.keys())

    print(f"Unique contexts: {len(contexts)}")

    # --------------------------------------------------
    # 3. 随机打乱 context
    # --------------------------------------------------
    random.shuffle(contexts)

    # --------------------------------------------------
    # 4. 根据 sample 数量决定 train / val
    # --------------------------------------------------
    target_train_size = int(len(data) * args.train_ratio)

    train = []
    val = []

    train_size = 0

    for context in contexts:
        samples = context_to_samples[context]

        # 如果加入这个 context 后不会超过目标 train size，
        # 就放到 train
        if train_size + len(samples) <= target_train_size:
            train.extend(samples)
            train_size += len(samples)
        else:
            val.extend(samples)

    # --------------------------------------------------
    # 5. 再次检查 context 是否发生泄漏
    # --------------------------------------------------
    train_contexts = {item["context"] for item in train}
    val_contexts = {item["context"] for item in val}

    overlap = train_contexts & val_contexts

    if overlap:
        raise RuntimeError(
            f"Context leakage detected! "
            f"{len(overlap)} contexts appear in both train and val."
        )

    # --------------------------------------------------
    # 6. 打乱 train / val 中的样本
    # --------------------------------------------------
    random.shuffle(train)
    random.shuffle(val)

    # --------------------------------------------------
    # 7. 创建输出目录
    # --------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")

    # --------------------------------------------------
    # 8. 保存
    # --------------------------------------------------
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # --------------------------------------------------
    # 9. 输出统计信息
    # --------------------------------------------------
    print(f"Train: {len(train)} samples")
    print(f"Val:   {len(val)} samples")
    print(f"Train ratio: {len(train) / len(data):.4f}")
    print(f"Unique train contexts: {len(train_contexts)}")
    print(f"Unique val contexts:   {len(val_contexts)}")
    print(f"Context overlap:       {len(overlap)}")
    print()
    print(f"Saved train to: {train_path}")
    print(f"Saved val to:   {val_path}")


if __name__ == "__main__":
    main()