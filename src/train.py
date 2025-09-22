import argparse
from pathlib import Path
from math import ceil

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

from lib.dataset import load_custom_dataset, slice_dataset


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.train_log = open(Path(log_path)/"train_loss.log", "w")
        self.eval_log = open(Path(log_path)/"eval_loss.log", "w")

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_log.write("step\ttrain_loss\n")
        self.eval_log.write("step\teval_loss\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_log.write(f"{state.global_step}\t{logs['loss']}\n")
        if "eval_loss" in logs:
            self.eval_log.write(f"{state.global_step}\t{logs['eval_loss']}\n")

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
        '--init-from', '-if', dest='init_from', choices=["config", "pre"],
        help='Initialize model from config or pre-trained model.'
    )
    parser.add_argument(
        '--config-name', '-cn', dest='config_name', type=str, required=False, default=None,
        help='Limit the number of samples to process.'
    )
    parser.add_argument(
        '--pre-model', '-pm', dest='pre_model', type=str, required=False, default=None,
        help='Path to pre-trained model'
    )
    parser.add_argument(
        '--checkpoint', '-cp', dest='checkpoint', type=str, required=False, default=None,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--epochs', '-e', dest='epochs', type=int, required=False, default=3,
        help='Path to pre-trained model'
    )
    parser.add_argument(
        '--data-limit', '-dl', dest='data_limit', type=int, required=False, default=0,
        help='Path to pre-trained model'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def model_config(model_name: str) -> GPT2Config | None:
    if model_name == "gpt-large":
        return GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )

    if model_name == "gpt-medium":
        return GPT2Config(
            vocab_size=50257,
            n_positions=512,
            n_embd=384,
            n_layer=6,
            n_head=6,
        )

    if model_name == "gpt-mini":
        return GPT2Config(
            vocab_size=50257,
            n_positions=128,
            n_embd=256,
            n_layer=4,
            n_head=4,
        )


def load_model_from_config(config_name: str) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(model_config(config_name))


def load_model_from_pre_trained(model_path: str) -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(model_path, ignore_mismatched_sizes=True)


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    args = read_args()
    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    # === Load model
    model: GPT2LMHeadModel | None = None
    if args.init_from == "config":
        print("Loading model from config:", args.config_name)
        model = load_model_from_config(args.config_name)
    elif args.init_from == "pre":
        print("Loading pre-trained model from:", args.pre_model)
        model = load_model_from_pre_trained(args.pre_model)
    assert model is not None
    model.save_pretrained(Path(args.out_path) / "init_model")

    # === load data
    dataset = load_custom_dataset(args.data_path, None, "local")

    if isinstance(dataset, Dataset):
        if args.data_limit > 0:
            data_limit = ceil(args.data_limit * 1.1)
            dataset = slice_dataset(dataset, 0, data_limit)
        data_dict = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    elif isinstance(dataset, DatasetDict):
        data_dict = dataset
        for k, dt in data_dict.items():
            if k == "train":
                if args.data_limit > 0:
                    data_dict[k] = slice_dataset(dt, 0, args.data_limit)
            else:
                data_dict[k] = slice_dataset(dt, 0, min(1000, args.data_limit))
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
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 4
    steps_per_epoch = train_dataset_size // (per_device_train_batch_size * gradient_accumulation_steps)
    # save_steps = steps_per_epoch // torch.cuda.device_count() // 3   # save on every 0.3 epoch
    # logging_steps = steps_per_epoch * args.epochs // 100 // torch.cuda.device_count()
    save_steps = steps_per_epoch // 10   # 0.1 epoch
    logging_steps = steps_per_epoch // 10
    # === train model
    training_args = TrainingArguments(
        output_dir=args.out_path,
        save_safetensors=True,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[LossLoggerCallback(f"{args.out_path}/logs")],
    )

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
