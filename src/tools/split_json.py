import sys
import json
from pathlib import Path
import random


def split_list(data, train_ratio=0.8, seed=42):
    # 复制一份原数据，避免修改原始列表
    temp_list = data.copy()

    # 设置随机种子
    random.seed(seed)

    # 打乱列表
    random.shuffle(temp_list)

    # 计算切分点
    split_index = int(len(temp_list) * train_ratio)

    # 切分
    part_1 = temp_list[:split_index]
    part_2 = temp_list[split_index:]

    return part_1, part_2


input_path = Path(sys.argv[1])
with open(input_path, "r") as f:
    data = json.load(f)
train_data, val_data = split_list(data, train_ratio=0.8, seed=42)   

output_path = Path(sys.argv[2])
output_path.mkdir(parents=True, exist_ok=True)
with open(output_path/"train.json", "w") as f:
    json.dump(train_data, f, indent=4)
with open(output_path/"test.json", "w") as f:
    json.dump(val_data, f, indent=4)
