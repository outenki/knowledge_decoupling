from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import torch
from transformers.training_args import TrainingArguments


def get_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple M1/M2
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def print_args(args: dict):
    print("↓↓↓↓↓↓↓↓↓↓ Arguments ↓↓↓↓↓↓↓↓↓↓")
    for k, v in args.items():
        print(f"{k}: {v}")
    print("↑↑↑↑↑↑↑↑↑↑ Arguments ↑↑↑↑↑↑↑↑↑↑") 
    print()


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


def inspect_checkpoint(cp_path: str) -> dict:
    info = {
        "exists": False,
        "is_dir": False,
        "files": [],
        "has_model": False,
        "has_optimizer": False,
        "has_scheduler": False,
        "has_trainer_state": False,
    }
    if cp_path is None:
        return info
    p = Path(cp_path)
    if not p.exists():
        return info
    info["exists"] = True
    info["is_dir"] = p.is_dir()
    try:
        if p.is_dir():
            info["files"] = [f.name for f in p.iterdir()]
        else:
            info["files"] = [p.name]
    except Exception:
        info["files"] = []
    fnames = set(info["files"])
    info["has_model"] = any(n in fnames for n in ("pytorch_model.bin", "pytorch_model.pt", "tf_model.h5", "model.safetensors")) or (p / "pytorch_model.bin").exists()
    info["has_optimizer"] = "optimizer.pt" in fnames or (p / "optimizer.pt").exists()
    info["has_scheduler"] = "scheduler.pt" in fnames or (p / "scheduler.pt").exists()
    info["has_trainer_state"] = "trainer_state.json" in fnames or (p / "trainer_state.json").exists()
    return info


def load_checkpoint_auxiliary_files(cp_path: str) -> dict:
    optimizer = None
    scheduler = None
    try:
        try:
            opt_path = Path(cp_path) / "optimizer.pt"
            optimizer = torch.load(opt_path, map_location="cpu")
        except Exception as e:
            print(f"‼️ Failed to load optimizer state: {e}")
        try:
            sch_path = Path(cp_path) / "scheduler.pt"
            scheduler = torch.load(sch_path, map_location="cpu")
            print(">>> Scheduler state loaded from checkpoint")
        except Exception as e:
            print(f"‼️ Failed to load scheduler state: {e}")
    except Exception as e:
        print(f"‼️ Error while inspecting/loading checkpoint auxiliary files: {e}")
    assert optimizer is not None or scheduler is not None, "Neither optimizer nor scheduler state could be loaded from checkpoint."
    return {"optimizer": optimizer, "scheduler": scheduler}