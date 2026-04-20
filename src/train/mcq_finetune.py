import argparse
import json
import math
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments


from src.lib.dataset import load_custom_dataset
from src.lib.utils import print_args


DECAY_RATE = 0.9  # for WSD


class MCQDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = torch.tensor([f["labels"] for f in features])
        option_start_ids = [f.get("option_start_ids") for f in features]

        max_len = max(len(choice) for f in input_ids for choice in f)

        padded = []
        for f in input_ids:
            choices = []
            for choice in f:
                pad_len = max_len - len(choice)
                choices.append(choice + [self.pad_token_id] * pad_len)
            padded.append(choices)

        batch = {
            "input_ids": torch.tensor(padded),  # (B, C, T)
            "labels": labels
        }
        if all(starts is not None for starts in option_start_ids):
            batch["option_start_ids"] = torch.tensor(option_start_ids)
        return batch


class MCQTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None,return_outputs=False):
        input_ids = inputs["input_ids"]  # (B, C, T)
        labels = inputs["labels"]
        option_start_ids = inputs.get("option_start_ids")

        B, C, T = input_ids.shape
        model_config = getattr(model, "config", None)
        if model_config is None and hasattr(model, "module"):
            model_config = getattr(model.module, "config", None)
        pad_id = model_config.pad_token_id

        input_ids = input_ids.view(B * C, T)
        if option_start_ids is not None:
            option_start_ids = option_start_ids.view(B * C)

        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        token_loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        ).view(B * C, -1)

        # By default, score all non-padding tokens. If option boundaries are
        # available, only score tokens belonging to the option span.
        mask = (shift_labels != pad_id).float()
        if option_start_ids is not None:
            target_positions = torch.arange(1, T, device=input_ids.device).unsqueeze(0)
            option_mask = target_positions >= option_start_ids.unsqueeze(1)
            mask = mask * option_mask.float()

        valid_tokens = mask.sum(dim=1).clamp_min(1.0)
        seq_loss = (token_loss * mask).sum(dim=1) / valid_tokens

        seq_loss = seq_loss.view(B, C)

        choice_logits = -seq_loss

        loss = torch.nn.CrossEntropyLoss()(choice_logits, labels)

        if return_outputs:
            return loss, {"logits": choice_logits}
        return loss


def training_args_to_dict(args: TrainingArguments) -> dict:
    serializable_args = {}
    args_dict = asdict(args) if is_dataclass(args) else vars(args)
    for k, v in args_dict.items():
        try:
            json.dumps(v)
            serializable_args[k] = v
        except (TypeError, OverflowError):
            serializable_args[k] = str(v)
    return serializable_args


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
        '--learning-rate', '-lr', dest='lr', type=float, required=False, default=5e-5,
    )
    parser.add_argument(
        '--epoch-checkpoints-num', '-ckn', dest='epoch_checkpoints_num', type=int, required=False, default=50,
    )
    parser.add_argument(
        '--save-checkpoints-num', '-sckn', dest='save_checkpoints_num', type=int, required=False, default=2,
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


def load_model_from_config(config_name: str, speedup: bool) -> Any:
    if speedup:
        # torch.backends.cuda.enable_flash_sdp(True)
        # torch.backends.cuda.enable_mem_efficient_sdp(True)
        # torch.backends.cuda.enable_math_sdp(True)
        # print(">>> speed up with torch 2.0 compile...")
        # model = torch.compile(model)
        print(">>> Applying speedup options...")
        print("- speed up with flash attention 3")
        return AutoModelForCausalLM.from_pretrained(
            config_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_3"
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            config_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )


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
            # For validation and test sets, use a smaller limit
            _data_limit = data_limit // 10
            data_dict[k] = random_sample(dt, _data_limit)
    return data_dict


def load_dataset_dict(data_path: str) -> DatasetDict:
    dataset = load_custom_dataset(data_path, None, "local")
    if isinstance(dataset, Dataset):
        data_dict = DatasetDict({"train": dataset})
    elif isinstance(dataset, DatasetDict):
        data_dict = dataset
    else:
        raise TypeError(f"Invalid data type {type(dataset)}")
    return data_dict

def load_and_limit_dataset_dict(data_path: str, data_limit: int) -> DatasetDict:
    """
    Load data and limit the number of samples for training.
    For validation and test sets, use a smaller limit (data_limit // 10).
    """
    print(f">>> Loading dataset from {data_path}")
    _data_dict = load_dataset_dict(data_path)
    print(_data_dict)
    print(">>> data loaded")
    if data_limit > 0:
        _data_dict = limit_dataset_dict(_data_dict, data_limit)
    return _data_dict


def main():
    print(">>> CUDA available:", torch.cuda.is_available())
    print(">>> GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f">>> GPU {i}: {torch.cuda.get_device_name(i)}")

    args = read_args()
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    print_args(vars(args))
    try:
        with open(Path(args.out_path)/"arguments.json", "w") as f:
            json.dump(vars(args), f, indent=4)
    except Exception as e:
        print(f"‼️Failed to dump arguments!")
        print(e)

    WANDB_RUN = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="tianqi-wang-a2-tohoku-university",
        group=args.config_name + datetime.now().strftime("-%Y%m%d"),
        # Set the wandb project where this run will be logged.
        project="Knowledge Decoupling",
    )
    WANDB_RUN.name = args.out_path.split("output/", maxsplit=1)[1]


    # === Load model
    model = None
    if args.init_model:
        print(">>> Loading init model from:", args.init_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.init_model,
            quantization_config=None,
        )
    else:
        print(">>> Loading model from config:", args.config_name)
        model = load_model_from_config(
            args.config_name,
            args.speedup
        )


    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 👈 先设！
    model.config.pad_token_id = tokenizer.pad_token_id

    assert model is not None
    model.save_pretrained(Path(args.out_path) / "init_model")

    # === load data
    data_list = []
    if args.data_limit is None:
        args.data_limit = [0] * len(args.data_path)
    assert len(args.data_path) == len(args.data_limit), "data_path and data_limit should have the same length."
    for dp, dl in zip(args.data_path, args.data_limit):
        _data_dict = load_and_limit_dataset_dict(dp, dl)
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

    # train_dataset.set_format("torch")
    # eval_dataset.set_format("torch")
    print(f">>> Training data size: {len(train_dataset)}")
    print(f">>> Eval data size: {len(eval_dataset)}")

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
    epoch_total_checkpoints = args.epoch_checkpoints_num
    save_steps = max(1, total_steps // (epoch_total_checkpoints * args.epochs))

    print(f">>> Total training steps: {total_steps}")
    print(f">>> Targeting {epoch_total_checkpoints} checkpoints.")
    print(f">>> Computed save/eval steps: {save_steps}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    training_args = TrainingArguments(
        output_dir=args.out_path,
        warmup_steps=warmup_steps,
        bf16=True,
        fp16=False,
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
        # optim="adamw_torch_fused",
    )

    # Save training arguments to JSON for later reference
    with open(Path(args.out_path) / "training_args.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)
    print(">>> eval_dataset:", eval_dataset)

    # WSD settings
    def ws_decay(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        wsd = DECAY_RATE ** ((step - warmup_steps) / steps_per_epoch)
        return cosine * wsd
    scheduler_ws = LambdaLR(optimizer, lr_lambda=ws_decay)
    trainer = MCQTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MCQDataCollator(tokenizer.pad_token_id),
        callbacks=[LossLoggerCallback(f"{args.out_path}/logs")],
        optimizers=(optimizer, scheduler_ws)
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

    # Load checkpoint if provided
    checkpoint = args.checkpoint
    if args.init_model and not checkpoint:
        print(f">>> Staring from model: {args.init_model}")
    elif not checkpoint or not Path(checkpoint).exists() or not Path(checkpoint).is_dir():
        checkpoint = None
        print(">>> Starting from hf model")
    else:
        print(f">>> Resuming from checkpoint: {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f">>> Save model to: {args.out_path}")
    model.save_pretrained(Path(args.out_path))
    WANDB_RUN.finish()


if __name__ == "__main__":
    main()
