import json
import math
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
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

from src.lib.dataset import load_custom_dataset
from src.lib.utils import print_args, training_args_to_dict, inspect_checkpoint, load_checkpoint_auxiliary_files
from src.lib.dataset import slice_dataset

DECAY_RATE = 0.9
random.seed(42)


def load_model_from_pretrained(model_path: str, speedup: bool) -> Any:
        print(">>> Loading init model from:", model_path)
        if speedup:
            print(">>> Applying speedup options...")
            print("- speed up with flash attention 3")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                # attn_implementation="flash_attention_3"
                attn_implementation="sdpa"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        print(">>> Init model loaded. Config:", model.config)
        return model


def load_model_from_config_random(config_name: str, speedup: bool) -> Any:
    print(">>> Initializing model from random weights using config:", config_name)
    config = AutoConfig.from_pretrained(config_name, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config

    if hasattr(config, "use_linear_attn"):
        config.use_linear_attn = False
    if hasattr(config, "attn_type"):
        config.attn_type = "sdpa"
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "sdpa"
    if speedup:
        print(">>> Applying speedup options...")
        # print("- speed up with flash attention 3")
        print("- speed up with sdpa")
        model = AutoModelForCausalLM.from_config(
            config,
            # torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # attn_implementation="flash_attention_3"
            attn_implementation="sdpa"
        )
    else:
        model = AutoModelForCausalLM.from_config(
            config,
            # torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    for name, module in model.named_modules():
        if hasattr(module, "use_linear_attn"):
            module.use_linear_attn = False
    model.tie_weights()
    # model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print(">>> Checking random model parameters...")
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() > 1:
            print(name, p.mean().item(), p.std().item())
            break
    return model.to(torch.bfloat16)
    

def _limit_dataset_dict(data_dict: DatasetDict, data_limit: int) -> DatasetDict:
    for k, dt in data_dict.items():
        dt = dt.shuffle(seed=42)
        if k == "train":
            data_dict[k] = slice_dataset(dt, 0, data_limit)
        else:
            # For validation and test sets, use a smaller limit
            _data_limit = data_limit // 10
            data_dict[k] = slice_dataset(dt, 0, _data_limit)
    return data_dict


def _load_dataset_dict(data_path: str) -> DatasetDict:
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
    _data_dict = _load_dataset_dict(data_path)
    print(_data_dict)
    print(">>> data loaded")
    if data_limit > 0:
        _data_dict = _limit_dataset_dict(_data_dict, data_limit)
    return _data_dict

@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train"
)
def main(cfg: DictConfig):

    # === Save a copy of the config for later reference
    Path(cfg.output.path).mkdir(parents=True, exist_ok=True)
    log_path = f"{cfg.output.path}/logs"
    Path(log_path).mkdir(parents=True, exist_ok=True)

    print_args(vars(cfg))
    try:
        with open(Path(cfg.output.path)/"arguments.json", "w") as f:
            json.dump(vars(cfg), f, indent=4)
    except Exception:
        print("‼️Failed to dumpt arguments!")

    print(">>> CUDA available:", torch.cuda.is_available())
    print(">>> GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f">>> GPU {i}: {torch.cuda.get_device_name(i)}")

    
    # === initialize wandb
    run_id_path = Path(cfg.output.path) / "wandb_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
    else:
        run_id = generate_id()
        run_id_path.write_text(run_id)
    WANDB_RUN = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="tianqi-wang-a2-tohoku-university",
        group=cfg.model.config + datetime.now().strftime("-%Y%m%d"),
        # Set the wandb project where this run will be logged.
        project="Knowledge Decoupling",
    )
    WANDB_RUN.name = cfg.output.path.split("output/", maxsplit=1)[1]


    # === Load model
    model = None
    if cfg.model.init_model:
        print(">>> Loading init model from pretrained model:", cfg.model.init_model)
        model = load_model_from_pretrained(cfg.model.init_model, cfg.speedup)
        print(">>> Init model loaded. Config:", model.config)
    else:
        assert cfg.model.config is not None, "--config-name is required when using --random-init"
        model = load_model_from_config_random(cfg.model.config, cfg.speedup)
        print(">>> Randomly initialized model. Config:", model.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.config if cfg.model.config is not None else cfg.model.init_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model is not None
    model.save_pretrained(Path(cfg.output.path) / "init_model")

    # === load data
    data_list = []
    if cfg.data.limit is None or len(cfg.data.limit) == 0:
        cfg.data.limit = [0] * len(cfg.data.paths)
    assert len(cfg.data.paths) == len(cfg.data.limit), "data_paths and data_limit should have the same length."
    for dp, dl in zip(cfg.data.paths, cfg.data.limit):
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
    for split in ("val", "validation", "dev"):
        if split in data_dict:
            eval_dataset = data_dict[split]
            break
    if eval_dataset is None:
        data_dict = train_dataset .train_test_split(test_size=0.05, shuffle=True, seed=42)
        train_dataset = data_dict["train"]
        eval_dataset = data_dict["test"]
    print(">>> dataset features:", train_dataset.features)
    print(f">>> Training data size: {len(train_dataset)}")
    print(f">>> Eval data size: {len(eval_dataset)}")


    # === Training setup
    train_dataset_size = len(train_dataset)
    per_device_train_batch_size = 2
    gradient_accumulation_steps = max(1, 16 // per_device_train_batch_size)
    world_size = max(1, torch.cuda.device_count())
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size

    # 计算总步数
    steps_per_epoch = train_dataset_size // effective_batch_size
    total_steps = steps_per_epoch * cfg.epochs

    # 计算 Warmup Steps
    warmup_steps = total_steps // 20 

    # 计算保存和评估间隔
    save_steps = max(1, total_steps // (cfg.checkpoints_per_epoch * cfg.epochs))

    print(f">>> Total training steps: {total_steps}")
    print(f">>> Targeting {cfg.checkpoints_per_epoch} checkpoints per epoch.")
    print(f">>> Computed save/eval steps: {save_steps}")


    training_args = TrainingArguments(
        output_dir=cfg.output.path,
        warmup_steps=warmup_steps,
        ddp_find_unused_parameters=False,
        bf16=True,
        dataloader_num_workers=4,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=cfg.epochs,
        logging_dir=log_path,
        logging_steps=save_steps,
        eval_steps=save_steps,
        save_steps=save_steps,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy=cfg.eval_strategy,
        report_to="wandb",
        save_total_limit=cfg.save_checkpoints_num,
    )

    # Save training arguments to JSON for later reference
    with open(Path(cfg.output.path) / "training_args.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)
    print(">>> eval_dataset:", eval_dataset)


    # WSD settings
    # optimizer = AdamW8bit(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    def ws_decay(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        wsd = DECAY_RATE ** ((step - warmup_steps) / steps_per_epoch)
        return cosine * wsd
    scheduler_ws = LambdaLR(optimizer, lr_lambda=ws_decay)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler_ws)
    )
    WANDB_RUN.config.update(trainer.args)
    WANDB_RUN.config.update({
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "effective_batch_size": effective_batch_size,
        "save_steps": save_steps,
        "batch_size": per_device_train_batch_size,
    })

    with open(Path(cfg.output.path) / "trainer.json", "w") as f:
        json.dump(trainer.args, f, indent=4)

    # Load checkpoint if provided
    checkpoint = cfg.model.checkpoint

    if not checkpoint:
        checkpoint = None
        print(">>> Starting from random model")
    else:
        info = inspect_checkpoint(checkpoint)
        assert info["exists"] and info["is_dir"] and info["has_model"] and info["has_optimizer"] and info["has_scheduler"] and info["has_trainer_state"]
        ck_state = load_checkpoint_auxiliary_files(checkpoint)
        optimizer = ck_state["optimizer"]
        scheduler = ck_state["scheduler"]
        trainer.optimizer = optimizer
        trainer.lr_scheduler = scheduler
        print(f">>> Resuming from checkpoint: {checkpoint}")
        # print(f">>> Resuming from checkpoint: {checkpoint}")
        # ck_info = inspect_checkpoint(checkpoint)
        # print(f">>> Checkpoint info: exists={ck_info['exists']}, is_dir={ck_info['is_dir']}, files={ck_info['files']}")
        # Try to load optimizer/scheduler state if available. Use CPU map_location to avoid device mismatch.

        # # If a raw model state dict exists, report mismatches when trying a non-strict load.
        # try:
        #     model_state_path = Path(checkpoint) / "pytorch_model.bin"
        #     if model_state_path.exists():
        #         print(">>> Found raw model weights in checkpoint; attempting a non-strict load to report mismatches.")
        #         state_dict = torch.load(model_state_path, map_location="cpu")
        #         # strip possible 'module.' prefix
        #         new_state = {}
        #         for k, v in state_dict.items():
        #             nk = k[len("module."):] if k.startswith("module.") else k
        #             new_state[nk] = v
        #         try:
        #             load_res = model.load_state_dict(new_state, strict=False)
        #             missing = getattr(load_res, "missing_keys", None) or load_res.get("missing_keys") if isinstance(load_res, dict) else []
        #             unexpected = getattr(load_res, "unexpected_keys", None) or load_res.get("unexpected_keys") if isinstance(load_res, dict) else []
        #             print(f">>> Model load (non-strict) reported missing keys: {missing}")
        #             print(f">>> Model load (non-strict) reported unexpected keys: {unexpected}")
        #         except Exception as e:
        #             print(f"‼️ Failed to load model state dict from checkpoint (non-strict): {e}")
        # except Exception as e:
        #     print(f"‼️ Error while inspecting/loading model weights: {e}")
    # Use Trainer's resume functionality. Pass the checkpoint path (or None) so Trainer can restore trainer state.
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f">>> Save model to: {cfg.output.path}")
    model.save_pretrained(Path(cfg.output.path))
    WANDB_RUN.finish()


if __name__ == "__main__":
    main()
