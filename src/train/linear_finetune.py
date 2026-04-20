from datetime import datetime
import argparse
from pathlib import Path
import json
from dataclasses import asdict, is_dataclass
import random

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from datasets import concatenate_datasets

from src.lib.dataset import load_custom_dataset
from src.lib.utils import get_device, print_args
from src.lib.linear_model import MCQModel, MCQCollator


def training_args_to_dict(args) -> dict:
    serializable_args = {}
    args_dict = asdict(args) if is_dataclass(args) else vars(args)
    for k, v in args_dict.items():
        try:
            json.dumps(v)
            serializable_args[k] = v
        except (TypeError, OverflowError):
            serializable_args[k] = str(v)
    return serializable_args


def load_dataset_dict(data_path: str) -> DatasetDict:
    dataset = load_custom_dataset(data_path, None, "local")
    if isinstance(dataset, Dataset):
        data_dict = DatasetDict({
            "train": dataset,
        })
    elif isinstance(dataset, DatasetDict):
        data_dict = dataset
    else:
        raise TypeError(f"Invalid data type {type(dataset)}")
    return data_dict




def load_model_from_config(config_name: str) -> AutoModel:
    if args.speedup:
        # print("- speed up with xformers")
        # torch.backends.cuda.enable_flash_sdp(True)
        # torch.backends.cuda.enable_mem_efficient_sdp(True)
        # torch.backends.cuda.enable_math_sdp(True)

        # print("- speed up with flash attention 3")

        # speed up with torch 2.0 compile
        # model = torch.compile(model)
        print(">>> Applying speedup options...")
        print("- speed up with flash attention 3")
        return AutoModel.from_pretrained(
            config_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_hidden_states=True,
            attn_implementation="flash_attention_3"
        )
    else:
        return AutoModel.from_pretrained(
            config_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', '-dp', dest='data_path', type=str, action='append',
        help='Dataset path to load from.'
    )
    parser.add_argument(
        '--config-name', '-cn', dest='config_name', type=str, required=False, default=None,
        help="Config name of models."
    )
    parser.add_argument(
        '--init-model', '-im', dest='init_model', type=str, required=False, default=None,
        help='Path to pre-trained model'
    )
    parser.add_argument(
        '--checkpoint', '-cp', dest='checkpoint', type=str, required=False, default=None,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--epochs', '-e', dest='epochs', type=int, required=False, default=3,
    )
    parser.add_argument(
        '--save-checkpoints', '-sc', dest='save_checkpoints', type=int, required=False, default=50,
    )
    parser.add_argument(
        '--data-limit', '-dl', dest='data_limit', type=int, action='append',
        help='Max number of samples for training. 0 for no limit.'
    )
    parser.add_argument(
        '--speedup', '-su', dest='speedup', action='store_true',
        help='Enable speedup options.'
    )
    parser.add_argument(
        '--custom-pad', '-pad', dest='custom_pad', action='store_true',
        help='Add custom PAD token'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()

def random_sample(dataset, number: int) -> Dataset:
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Invalid data type {type(dataset)}")
    if number >= len(dataset):
        return dataset
    indices = random.sample(range(len(dataset)), number)
    return dataset.select(indices)

def limit_dataset_dict(data_dict: DatasetDict, data_limit: int) -> DatasetDict:
    for k, dt in data_dict.items():
        if k == "train":
            data_dict[k] = random_sample(dt, data_limit)
        else:
            data_limit = data_limit // 10
            data_dict[k] = random_sample(dt, data_limit)
    return data_dict

def main():
    print(">>> CUDA available:", torch.cuda.is_available())
    print(">>> GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f">>> GPU {i}: {torch.cuda.get_device_name(i)}")

    args = read_args()
    print_args(vars(args))

    WANDB_RUN = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="tianqi-wang-a2-tohoku-university",
        group=args.config_name + datetime.now().strftime("-%Y%m%d"),
        # Set the wandb project where this run will be logged.
        project="Knowledge Decoupling",
    )
    WANDB_RUN.name = args.out_path.split("output/", maxsplit=1)[1]
    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    # === save arguments as json
    try:
        with open(Path(args.out_path)/"arguments.json", "w") as f:
            json.dump(vars(args), f, indent=4)
    except Exception:
        print(f"‼️Failed to dumpt arguments!")

    if args.init_model:
        print(">>> Loading init model from:", args.init_model)
        init_model = AutoModel.from_pretrained(
            args.init_model,
            quantization_config=None,
        )
    else:
        init_model = load_model_from_config(
            args.config_name,
            args.speedup
        )
    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    if args.custom_pad:
        print(">>> Adding custom PAD token to the tokenizer and model.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        init_model.resize_token_embeddings(len(tokenizer))
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # 👈 先设！
    init_model.config.pad_token_id = tokenizer.pad_token_id


    # === load data
    data_list = []
    if args.data_limit is None:
        args.data_limit = [0] * len(args.data_path)
    assert len(args.data_path) == len(args.data_limit), "data_path and data_limit should have the same length."
    for data_path, data_limit in zip(args.data_path, args.data_limit):
        print(f">>> Loading dataset from {data_path}")
        _data_dict = load_dataset_dict(data_path)
        print(_data_dict)
        print(">>> data loaded")
        if data_limit > 0:
            _data_dict = limit_dataset_dict(_data_dict, data_limit)
        # for k in _data_dict.keys():
        #     _data_dict[k] = normalize_dataset(_data_dict[k])
        data_list.append(_data_dict)
    data_dict = DatasetDict({
        split: concatenate_datasets([dd[split] for dd in data_list])
        for split in data_list[0].keys()
    })

    print(">>> Dataset:", data_dict)

    train_dataset: Dataset | None = None
    eval_dataset: Dataset | None = None
    train_dataset = data_dict["train"]
    if "val" in data_dict:
        eval_dataset = data_dict["val"]
    elif "validation" in data_dict:
        eval_dataset = data_dict["validation"]
    else:
        data_dict = train_dataset .train_test_split(test_size=0.05, shuffle=True, seed=42)
        train_dataset = data_dict["train"]
        eval_dataset = data_dict["test"]

    print(">>> dataset features:", train_dataset.features)

    print(f">>> Training data size: {len(train_dataset)}")
    print(f">>> Eval data size: {len(eval_dataset)}")

    num_choices = len(train_dataset[0]["input_ids"])

    model = MCQModel(init_model, num_choices)
    log_path = f"{args.out_path}/logs"
    Path(log_path).mkdir(parents=True, exist_ok=True)

    train_dataset_size = len(train_dataset)
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 16
    world_size = max(1, torch.cuda.device_count())
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size

    # 计算总步数
    steps_per_epoch = train_dataset_size // effective_batch_size
    total_steps = steps_per_epoch * args.epochs

    # 计算 Warmup Steps
    warmup_steps = total_steps // 20 

    # 计算保存和评估间隔
    target_total_checkpoints = args.save_checkpoints
    save_steps = max(1, total_steps // (target_total_checkpoints * args.epochs))

    print(f">>> Total training steps: {total_steps}")
    print(f">>> Targeting {target_total_checkpoints} checkpoints.")
    print(f">>> Computed save/eval steps: {save_steps}")

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    training_args = TrainingArguments(
        output_dir=args.out_path,
        warmup_steps=warmup_steps,
        remove_unused_columns=False,
        bf16=True,
        fp16=False,  # use bf16 instead of fp16
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=args.epochs,
        logging_dir=log_path,
        logging_steps=save_steps,
        eval_steps=save_steps,
        save_steps=save_steps,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy="steps",
        report_to="wandb",
        save_total_limit=2,
        # speed up with fused AdamW optimizer
        optim="adamw_torch_fused",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MCQCollator(tokenizer.pad_token_id),
    )

    WANDB_RUN.config.update(training_args_to_dict(trainer.args))
    WANDB_RUN.config.update({
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "effective_batch_size": effective_batch_size,
        "save_steps": save_steps,
        "batch_size": per_device_train_batch_size,
    })

    with open(Path(args.out_path) / "trainer.json", "w") as f:
        json.dump(training_args_to_dict(trainer.args), f, indent=4)

    trainer.train()

    print(f">>> Save model to: {args.out_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_choices": model.num_choices,
        },
        f"{args.out_path}/model.pt"
    )
    model.model.save_pretrained(args.out_path)
    tokenizer.save_pretrained(args.out_path)
    WANDB_RUN.finish()


if __name__ == "__main__":
    main()