import argparse
from pathlib import Path
from math import ceil
from functools import partial

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
        help='Model configs.'
    )
    parser.add_argument(
        '--pre-model', '-pm', dest='pre_model', type=str, required=False, default=None,
        help='Path to pre-trained model'
    )
    parser.add_argument(
        '--data-limit', '-dl', dest='data_limit', type=int, required=False, default=0,
        help='Limit the size of training data.'
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
    return GPT2LMHeadModel.from_pretrained(model_path)


def _chunk_input_ids(examples, block_size):

    input_ids = []
    for ids in examples["input_ids"]:
        input_ids += ids
        
    total_length = len(input_ids)

    # We drop the last chunk if it's smaller than block_size
    total_length = (total_length // block_size) * block_size

    result = {}
    result["input_ids"] = [
        input_ids[i: i + block_size]
        for i in range(0, total_length, block_size)
    ]
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    args = read_args()
    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    # === Load model
    model: GPT2LMHeadModel | None = None
    if args.init_from == "config":
        model = load_model_from_config(args.config_name)
    elif args.init_from == "pre":
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
                data_dict[k] = slice_dataset(dt, 0, args.data_limit)
            else:
                data_dict[k] = slice_dataset(dt, 0, 1000)
    else:
        raise TypeError(f"Invalid data type {type(dataset)}")

    train_dataset = data_dict["train"]
    eval_dataset = data_dict.get("val") or data_dict.get("eval") or data_dict.get("test")
    if eval_dataset is None:
        data_dict = train_dataset .train_test_split(test_size=0.01, shuffle=True, seed=42)
        train_dataset = data_dict["train"]
        eval_dataset = data_dict["test"]

    _chunk_fn = partial(_chunk_input_ids, max_length=model.config.n_positions)
    # === sliding window
    train_dataset = train_dataset.map(
        _chunk_input_ids,
        batched=True,
        batch_size=1000,
        remove_columns=["input_ids"],
        fn_kwargs={"block_size": model.config.n_positions}
    )
    eval_dataset = eval_dataset.map(
        _chunk_input_ids,
        batched=True,
        batch_size=1000,
        remove_columns=["input_ids"],
        fn_kwargs={"block_size": model.config.n_positions}
    )

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    print(f"Training data size: {len(train_dataset)}")
    print(f"Eval data size: {len(eval_dataset)}")

    log_path = f"{args.out_path}/logs"
    Path(log_path).mkdir(parents=True, exist_ok=True)
    # === train model
    training_args = TrainingArguments(
        output_dir=args.out_path,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir=log_path,
        logging_steps=100,
        eval_steps=100,
        logging_strategy="steps",
        save_strategy="no",
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

    trainer.train()
    model.save_pretrained(Path(args.out_path))


if __name__ == "__main__":
    main()
