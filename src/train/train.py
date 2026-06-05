import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from wandb.sdk.lib.runid import generate_id
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
# from bitsandbytes.optim import AdamW8bit
from bitsandbytes.optim.adamw import AdamW

from src.lib.dataset import load_dataset_for_training
from src.lib.utils import inspect_checkpoint
from src.lib.trainer import train_model_with_data, init_wandb_run
from src.lib.model import load_model_from_pretrained, load_model_from_config_random

DECAY_RATE = 0.9
random.seed(42)


@hydra.main(
    version_base=None,
    config_path="../../config/train",
    config_name="_"
)
def main(cfg: DictConfig):
    # === Save a copy of the config for later reference
    Path(cfg.output.path).mkdir(parents=True, exist_ok=True)
    log_path = f"{cfg.output.path}/logs"
    Path(log_path).mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))
    with open(Path(cfg.output.path) / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print(">>> CUDA available:", torch.cuda.is_available())
    print(">>> GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f">>> GPU {i}: {torch.cuda.get_device_name(i)}")

    
    # === Load model
    model = None
    if cfg.model.init_model:
        print(">>> Loading init model from pretrained model:", cfg.model.init_model)
        model = load_model_from_pretrained(cfg.model.init_model, cfg.runtime.attn_implementation)
        print(">>> Init model loaded. Config:", model.config)
    else:
        assert cfg.model.config is not None, "--config-name is required when using --random-init"
        model = load_model_from_config_random(cfg.model.config, cfg.runtime.attn_implementation)
        print(">>> Randomly initialized model. Config:", model.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.config if cfg.model.config is not None else cfg.model.init_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model is not None
    model.save_pretrained(Path(cfg.output.path) / "init_model")

    # === train model
    train_dataset, eval_dataset = load_dataset_for_training(cfg)
    wandb_run = init_wandb_run(cfg)
    train_model_with_data(model, train_dataset, eval_dataset, cfg, wandb_run) 
    wandb_run.finish()


if __name__ == "__main__":
    main()
