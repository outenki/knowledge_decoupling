import argparse
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import torch

from lib.dataset import load_custom_dataset


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.train_log = open(Path(log_path)/"train_loss.log", "w")
        self.eval_log =  open(Path(log_path)/"eval_loss.log", "w")

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_log.write(f"step\ttrain_loss\n")
        self.eval_log.write(f"step\teval_loss\n")

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
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def model_config(model_name: str) -> GPT2Config | None:
    if model_name == "gpt-mini":
        return GPT2Config(
            vocab_size=50257,
            n_positions=128,
            n_ctx=128,
            n_embd=256,
            n_layer=4,
            n_head=4,
        )


def load_model_from_config(config_name: str) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(model_config(config_name))


def load_model_from_pre_trained(model_path: str) -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(model_path)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple M1/M2
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


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
    data = load_custom_dataset(args.data_path, None, "local")

    if isinstance(data, Dataset):
        data_dict = data.train_test_split(test_size=0.1, shuffle=True, seed=42)
    elif isinstance(data, DatasetDict):
        data_dict = data
    else:
        raise TypeError(f"Invalid data type {type(data)}")

    train_dataset: Dataset | None = None
    eval_dataset: Dataset | None = None
    train_dataset = data_dict["train"]
    if "val" in data_dict:
        eval_dataset = data_dict["val"]
    elif "eval" in data_dict:
        eval_dataset = data_dict["eval"]
    else:
        eval_dataset = data_dict["test"]
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # === train model
    training_args = TrainingArguments(
        output_dir=args.out_path,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir=f"{args.out_path}/logs",
        logging_steps=500,
        eval_steps=500,
        save_steps=500,
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

    trainer.train()
    model.save_pretrained(Path(args.out_path))


if __name__ == "__main__":
    main()
