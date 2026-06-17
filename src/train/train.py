import random
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer

from src.lib.dataset import load_dataset_for_training
from src.lib.trainer import train_model_with_data, init_wandb_run
from src.lib.model import load_model_from_pretrained, load_model_from_config_random
from src.lib.model import freeze_parameters

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

    print(OmegaConf.to_yaml(cfg))
    with open(Path(cfg.output.path) / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print(">>> CUDA available:", torch.cuda.is_available())
    print(">>> GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f">>> GPU {i}: {torch.cuda.get_device_name(i)}")

    if cfg.training.attn_implementation == "flash_attention_3":
        # Hack to make flash attention 3 work
        import transformers.modeling_utils as mu
        mu.FLASH_ATTENTION_COMPATIBILITY_MATRIX[3]["pkg_availability_check"] = lambda: True

    # === Load model
    model = None
    if cfg.model.init_model:
        print(">>> Loading init model from pretrained model:", cfg.model.init_model)
        model = load_model_from_pretrained(cfg.model.init_model, cfg.training.attn_implementation)
        print(">>> Init model loaded. Config:", model.config)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.init_model)
    else:
        assert cfg.model.config is not None, "model.config is required"
        model = load_model_from_config_random(cfg.model.config, cfg.training.attn_implementation)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.config)
        print(">>> Randomly initialized model. Config:", model.config)
    
    if tokenizer.vocab_size is None or tokenizer.vocab_size == 0:
        print(">>> Tokenizer vocab size is 0 or None")
        print(f"Loading tokenizer from: {cfg.model.config}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model is not None
    model.save_pretrained(Path(cfg.output.path) / "init_model")
    tokenizer.save_pretrained(Path(cfg.output.path) / "init_model")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}, pad token id: {tokenizer.pad_token_id}")

    if cfg.model.freeze_layers != 0:
        freeze_parameters(model, cfg.model.freeze_layers)
        print(f">>> Frozen the bottom {cfg.model.freeze_layers} layers.")

    # === train model
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
