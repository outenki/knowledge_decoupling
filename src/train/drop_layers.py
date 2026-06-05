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
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer

from src.lib.dataset import load_dataset_for_training
from src.lib.model import load_model_from_pretrained, load_model_from_config_random
from src.lib.trainer import train_model_with_data, init_wandb_run

random.seed(42)


def get_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    raise NotImplementedError(
        f"Unknown architecture: {type(model)}"
    )

def get_num_layers(model):
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers

    if hasattr(model.config, "n_layer"):
        return model.config.n_layer

    raise NotImplementedError

def set_new_layers(model, new_layers):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = torch.nn.ModuleList(new_layers)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = torch.nn.ModuleList(new_layers)
    else:
        raise NotImplementedError  

def set_num_layers(model, layer_num):
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = layer_num
    elif hasattr(model.config, "n_layer"):
        model.config.n_layer = layer_num
    else:
        raise NotImplementedError

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_parameters(layer):
    for param in layer.parameters():
        param.requires_grad = True

def reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


@hydra.main(
    version_base=None,
    config_path="../../config/drop_layers",
    config_name="_"
)
def main(cfg: DictConfig):
    # save the config to the output directory for future reference
    Path(cfg.output.path).mkdir(parents=True, exist_ok=True)
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
        model = load_model_from_pretrained(cfg.model.init_model, cfg.training.attn_implementation)
        print(">>> Init model loaded. Config:", model.config)
    else:
        assert cfg.model.config is not None, "--config-name is required when using --random-init"
        model = load_model_from_config_random(cfg.model.config, cfg.training.attn_implementation)
        print(">>> Randomly initialized model. Config:", model.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.config if cfg.model.config is not None else cfg.model.init_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


    # === drop layers
    new_layers = None
    layers = get_layers(model)
    n_layers = len(layers)
    assert n_layers == get_num_layers(model), "Layer count mismatch"
    keep_n = cfg.model.keep_n_layers + 1  # add a new layer on top to finetune
    assert keep_n > 0 and keep_n <= n_layers, f"keep_n_layers={keep_n} is out of valid range. Valid range is 1 to {n_layers}."

    print(f">>> Original number of layers: {n_layers}")
    print(f">>> Keeping top {keep_n} layers and dropping the rest.")
    new_layers = layers[:keep_n]
    assert new_layers is not None, "New layers should have been set by this point."

    set_new_layers(model, new_layers)
    set_num_layers(model, keep_n)
    freeze_parameters(model)
    unfreeze_parameters(new_layers[-1])
    
    # init the new top layer with random weights
    if cfg.model.reset_top_layer:
        print(">>> Initializing the new top layer with random weights.")
        get_layers(model)[-1].apply(reset_parameters)


    if cfg.finetune:
        train_dataset, eval_dataset = load_dataset_for_training(cfg.data.paths, cfg.data.limits)
        wandb_run = init_wandb_run(cfg.output.path, cfg.model.config + datetime.now().strftime("-%Y%m%d"))
        model = train_model_with_data(
            model, train_dataset, eval_dataset, wandb_run,
            output_path=cfg.output.path,
            batch_size=cfg.training.batch_size,
            epochs=cfg.training.epochs,
            checkpoints_per_epoch=cfg.training.checkpoints_per_epoch,
            eval_strategy=cfg.training.eval_strategy,
            save_checkpoints_num=cfg.training.save_checkpoints_num,
            learning_rate=cfg.training.learning_rate,
            attn_implementation=cfg.training.attn_implementation,
            checkpoint=cfg.model.checkpoint
        ) 
        wandb_run.finish()
    print(f">>> Save model to: {cfg.output.path}")
    model.save_pretrained(Path(cfg.output.path))


if __name__ == "__main__":
    main()
