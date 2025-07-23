# %%
from datasets import Dataset
import spacy

# 示例数据
data = {
    "text": [
        "This is the first sentence. Here is the second one.",
        "Another paragraph with two sentences. And a third sentence here.",
    ]
}

# 创建 Dataset 对象
dataset = Dataset.from_dict(data)

# 加载 spaCy 模型（可选使用 en_core_web_sm 或 en_core_web_trf）
nlp = spacy.load("en_core_web_sm")

# 定义分句函数，返回一个 dict 包含所有句子
def split_into_sentences(example):
    doc = nlp(example["text"])
    return {"sentences": [sent.text for sent in doc.sents]}

# 使用 map 分割句子
dataset = dataset.map(split_into_sentences)

# 展平数据（将每一条的 sentence 列表展开成单独样本）
from itertools import chain

flattened_dataset = Dataset.from_dict({
    "sentence": list(chain.from_iterable(dataset["sentences"]))
})

# 查看结果
print(flattened_dataset)
# %%
flattened_dataset["sentence"][:5]  # 查看前5个句子
# %%
