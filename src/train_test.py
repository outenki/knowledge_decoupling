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


from lib.dataset import load_custom_dataset


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
        '--data-path', '-dp', dest='data_path', type=str,
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
        '--data-limit', '-dl', dest='data_limit', type=int, required=False, default=0,
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


def model_config(model_name: str) -> AutoConfig | None:
    # if model_name == "gpt2":
    #     return GPT2Config(
    #         vocab_size=50257,
    #         n_positions=1024,
    #         n_embd=768,
    #         n_layer=12,
    #         n_head=12,
    #     )

    # if model_name == "gpt-medium":
    #     return GPT2Config(
    #         vocab_size=50257,
    #         n_positions=512,
    #         n_embd=384,
    #         n_layer=6,
    #         n_head=6,
    #     )

    # if model_name == "gpt-mini":
    #     return GPT2Config(
    #         vocab_size=50257,
    #         n_positions=128,
    #         n_embd=256,
    #         n_layer=4,
    #         n_head=4,
    #    )
    return AutoConfig(model_name)


def load_model_from_config(config_name: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_config(model_config(config_name))


def random_sample(dataset, number: int) -> Dataset:
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Invalid data type {type(dataset)}")
    if number >= len(dataset):
        return dataset
    indices = random.sample(range(len(dataset)), number)
    return dataset.select(indices)


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    args = read_args()
    print(vars(args))
    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    # === Load model
    model: AutoModelForCausalLM | None = None
    print("Loading model from config:", args.config_name)
    model = load_model_from_config(args.config_name)
    if args.init_model:
        print("Loading model from local path:", args.init_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.init_model,
            quantization_config=None,
            device_map="auto",
        )
            


    if args.speedup:
        print("Applying speedup options...")
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
    dataset = load_custom_dataset(args.data_path, None, "local")

    if isinstance(dataset, Dataset):
        if args.data_limit > 0:
            data_limit = ceil(args.data_limit * 1.1)
            # dataset = slice_dataset(dataset, 0, data_limit)
            dataset = random_sample(dataset, data_limit)
        data_dict = dataset.train_test_split(train_size=1/1.1, shuffle=True, seed=42)
    elif isinstance(dataset, DatasetDict):
        data_dict = dataset
        for k, dt in data_dict.items():
            if args.data_limit > 0:
                if k == "train":
                    data_dict[k] = random_sample(dt, args.data_limit)
                else:
                    data_limit = args.data_limit // 10
                    data_dict[k] = random_sample(dt, data_limit)
    else:
        raise TypeError(f"Invalid data type {type(dataset)}")

    train_dataset: Dataset | None = None
    eval_dataset: Dataset | None = None
    train_dataset = data_dict["train"]
    if "val" in data_dict:
        eval_dataset = data_dict["val"]
    elif "validation" in data_dict:
        eval_dataset = data_dict["validation"]
    elif "eval" in data_dict:
        eval_dataset = data_dict["eval"]
    elif "test" in data_dict:
        eval_dataset = data_dict["test"]
    else:
        data_dict = train_dataset .train_test_split(test_size=0.01, shuffle=True, seed=42)
        train_dataset = data_dict["train"]
        eval_dataset = data_dict["test"]

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    print(f"Training data size: {len(train_dataset)}")
    print(f"Eval data size: {len(eval_dataset)}")

    log_path = f"{args.out_path}/logs"
    Path(log_path).mkdir(parents=True, exist_ok=True)

    train_dataset_size = len(train_dataset)
    per_device_train_batch_size = 16
    gradient_accumulation_steps = 16
    world_size = max(1, torch.cuda.device_count())

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = train_dataset_size // effective_batch_size

    save_steps = max(50, steps_per_epoch // 20)
    logging_steps = max(50, steps_per_epoch // 20)
    print("save steps:", save_steps)
    print("logging steps:", logging_steps)
    print("eval steps:", logging_steps)

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
        logging_steps=logging_steps,
        eval_steps=logging_steps,
        save_steps=save_steps,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy="steps",
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,

        # speed up with fused AdamW optimizer
        optim="adamw_torch_fused",
    )

    with open(Path(args.out_path) / "training_args.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)

    print("eval_dataset:", eval_dataset)
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
        print(f"Starting from random model")
    else:
        print(f"Resuming from checkpoint: {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f"Save model to: {args.out_path}")
    model.save_pretrained(Path(args.out_path))


if __name__ == "__main__":
    main()
