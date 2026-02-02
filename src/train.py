import argparse
from pathlib import Path
from math import ceil
import random
import json
from dataclasses import asdict, is_dataclass


import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets import concatenate_datasets

from lib.dataset import load_custom_dataset
from lib.utils import print_args


DECAY_RATE = 0.9  # for WSD


def training_args_to_dict(args) -> dict:
    serializable_args = {}
    args_dict = asdict(args) if is_dataclass(args) else vars(args)
    for k, v in args_dict.items():
        try:
            json.dumps(v)
            serializable_args[k] = v
        except (TypeError, OverflowError):
            serializable_args[k] = str(v)


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.train_log = open(Path(log_path)/"train_loss.log", "w")
        self.eval_log = open(Path(log_path)/"eval_loss.log", "w")

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_log.write("step\ttrain_loss\n")
        self.train_log.flush()
        self.eval_log.write("step\teval_loss\n")
        self.eval_log.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_log.write(f"{state.global_step}\t{logs['loss']}\n")
            self.train_log.flush()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            # 评估日志使用 state.global_step，确保与训练步骤对齐
            self.eval_log.write(f"{state.global_step}\t{metrics['eval_loss']}\n")
            self.eval_log.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.train_log.close()
        self.eval_log.close()


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
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def model_config(model_name: str):
    return AutoConfig.from_pretrained(model_name)


def load_model_from_config(config_name: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_config(model_config(config_name))


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


def load_dataset_dict(data_path: str) -> DatasetDict:
    dataset = load_custom_dataset(data_path, None, "local")
    if isinstance(dataset, Dataset):
        total_len = len(dataset)
        test_size = min(max(20, total_len // 10), total_len // 2)
        if total_len < 2:
            raise ValueError(f"Dataset at {data_path} is too small to split.")
        data_dict = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    elif isinstance(dataset, DatasetDict):
        data_dict = dataset
    else:
        raise TypeError(f"Invalid data type {type(dataset)}")
    return data_dict


def normalize_dataset(ds):
    def fix(x):
        if "attention_mask" not in x or x["attention_mask"] is None:
            x["attention_mask"] = [1] * len(x["input_ids"])
        return x
    return ds.map(fix)


def main():
    print(">>> CUDA available:", torch.cuda.is_available())
    print(">>> GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f">>> GPU {i}: {torch.cuda.get_device_name(i)}")

    args = read_args()
    print_args(vars(args))
    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    # === Load model
    model = None
    print(">>> Loading model from config:", args.config_name)
    model = load_model_from_config(args.config_name)
    if args.init_model:
        print(">>> Loading init model from:", args.init_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.init_model,
            quantization_config=None,
            device_map="auto",
        )
            


    if args.speedup:
        print(">>> Applying speedup options...")
        # print("- speed up with xformers")
        # torch.backends.cuda.enable_flash_sdp(True)
        # torch.backends.cuda.enable_mem_efficient_sdp(True)
        # torch.backends.cuda.enable_math_sdp(True)

        # print("- speed up with flash attention 3")
        model.config.attn_implementation = "flash_attention_3"

        # speed up with torch 2.0 compile
        # model = torch.compile(model)

    assert model is not None
    model.save_pretrained(Path(args.out_path) / "init_model")

    # === load data
    data_list = []
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
        data_dict = train_dataset .train_test_split(test_size=0.01, shuffle=True, seed=42)
        train_dataset = data_dict["train"]
        eval_dataset = data_dict["test"]

    print(">>> dataset features:", train_dataset.features)


    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    print(f">>> Training data size: {len(train_dataset)}")
    print(f">>> Eval data size: {len(eval_dataset)}")

    log_path = f"{args.out_path}/logs"
    Path(log_path).mkdir(parents=True, exist_ok=True)

    train_dataset_size = len(train_dataset)
    per_device_train_batch_size = 16
    gradient_accumulation_steps = 16
    world_size = max(1, torch.cuda.device_count())

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size

    total_steps = train_dataset_size // effective_batch_size

    target_total_checkpoints = args.save_checkpoints
    save_steps = max(1, total_steps // target_total_checkpoints)

    print(f">>> Total training steps: {total_steps}")
    print(f">>> Targeting {target_total_checkpoints} checkpoints.")
    print(f">>> Computed save/eval steps: {save_steps}")

    training_args = TrainingArguments(
        output_dir=args.out_path,
        save_safetensors=True,
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
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        # speed up with fused AdamW optimizer
        optim="adamw_torch_fused",
    )

    with open(Path(args.out_path) / "training_args.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)

    print(">>> eval_dataset:", eval_dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[LossLoggerCallback(f"{args.out_path}/logs")],
    )

    # WSD settings
    total_steps = len(trainer.get_train_dataloader()) * training_args.num_train_epochs
    warmup_steps = training_args.warmup_steps or (total_steps // 20)
    optimizer = AdamW(model.parameters(), lr=5e-4)

    def ws_decay(step):
        # WSD
        if step < warmup_steps:
            # linear Warmup
            return step / warmup_steps

        # Cosine as the baseline
        cosine_ratio = max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.1415926535 * (step - warmup_steps) / (total_steps - warmup_steps)))))

        epoch_step = len(trainer.get_train_dataloader())
        wsd_factor = DECAY_RATE ** ((step - warmup_steps) / epoch_step)

        return cosine_ratio * wsd_factor

    scheduler_ws = LambdaLR(optimizer, lr_lambda=ws_decay)
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler_ws

    with open(Path(args.out_path) / "trainer.json", "w") as f:
        json.dump(training_args_to_dict(trainer.args), f, indent=4)

    # Load checkpoint if provided
    checkpoint = args.checkpoint
    if not checkpoint or not Path(checkpoint).exists() or not Path(checkpoint).is_dir():
        checkpoint = None
        print(f">>> Starting from random model")
    else:
        print(f">>> Resuming from checkpoint: {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f">>> Save model to: {args.out_path}")
    model.save_pretrained(Path(args.out_path))


if __name__ == "__main__":
    main()
