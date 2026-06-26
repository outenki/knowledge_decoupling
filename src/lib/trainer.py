import torch
import json
from pathlib import Path
import math

from transformers.trainer import Trainer
from torch.optim.lr_scheduler import LambdaLR
from transformers.training_args import TrainingArguments
from bitsandbytes.optim.adamw import AdamW
import wandb
from wandb.sdk.lib.runid import generate_id

from src.lib.utils import inspect_checkpoint
from src.lib.model import get_layers

DECAY_RATE = 0.9


def init_wandb_run(output_path, group_name):
    run_id_path = Path(output_path) / "wandb_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
    else:
        run_id = generate_id()
        run_id_path.write_text(run_id)
    wandb_run = wandb.init(
        id=run_id,
        resume="allow",
        dir=output_path,
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="tianqi-wang-a2-tohoku-university",
        group=group_name,  # Set the wandb group to organize runs together (e.g., by experiment or model).
        # Set the wandb project where this run will be logged.
        project="Knowledge Decoupling",
    )
    wandb_run.name = output_path.split("output/", maxsplit=1)[1]
    return wandb_run


def train_model_with_data(
    model, train_dataset, eval_dataset, wandb_run, output_path,
    batch_size, epochs, checkpoints_per_epoch, eval_strategy, save_checkpoints_num, learning_rate, attn_implementation, checkpoint=None
):
    # === Training setup
    log_path = f"{output_path}/logs"
    Path(log_path).mkdir(parents=True, exist_ok=True)
    train_dataset_size = len(train_dataset)
    per_device_train_batch_size = batch_size
    gradient_accumulation_steps = max(1, 16 // per_device_train_batch_size)
    world_size = max(1, torch.cuda.device_count())
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size

    # 计算总步数
    steps_per_epoch = train_dataset_size // effective_batch_size
    total_steps = steps_per_epoch * epochs

    # 计算 Warmup Steps
    warmup_steps = total_steps // 20 

    # 计算保存和评估间隔
    save_steps = max(1, total_steps // (checkpoints_per_epoch * epochs))
    eval_steps = save_steps // 5
    logging_steps = eval_steps

    print(f">>> Total training steps: {total_steps}")
    print(f">>> Targeting {checkpoints_per_epoch} checkpoints per epoch.")
    print(f">>> Computed save steps/times: {save_steps}/{total_steps // save_steps}")
    print(f">>> Computed eval/logging steps and times: {logging_steps}/{total_steps // logging_steps}")


    training_args = TrainingArguments(
        output_dir=output_path,
        warmup_steps=warmup_steps,
        ddp_find_unused_parameters=False,
        bf16=True,
        dataloader_num_workers=8,  # Increased from 4 for better data loading
        dataloader_prefetch_factor=2,  # Prefetch more batches
        dataloader_pin_memory=True,  # Pin memory for faster transfers
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=epochs,
        logging_dir=log_path,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy=eval_strategy,
        report_to="wandb",
        save_total_limit=save_checkpoints_num,
        optim="adamw_torch",  # Use optimized AdamW
        max_grad_norm=1.0,  # Gradient clipping
        gradient_checkpointing=True,  # Enable gradient checkpointing
    )

    # Save training arguments to JSON for later reference
    with open(Path(output_path) / "training_args.json", "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)
    print(">>> eval_dataset:", eval_dataset)


    # WSD settings
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
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
    wandb_run.config.update(trainer.args.to_dict(), allow_val_change=True)
    wandb_run.config.update({
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "effective_batch_size": effective_batch_size,
        "save_steps": save_steps,
        "batch_size": per_device_train_batch_size,
    }, allow_val_change=True)
    with open(Path(output_path) / "trainer.json", "w") as f:
        json.dump(trainer.args.to_dict(), f, indent=4)

    # Load checkpoint if provided
    checkpoint = checkpoint

    if not checkpoint:
        checkpoint = None
    else:
        info = inspect_checkpoint(checkpoint)
        assert info["exists"] and info["is_dir"] and info["has_model"] and info["has_optimizer"] and info["has_scheduler"] and info["has_trainer_state"]
        print(f">>> Resuming from checkpoint: {checkpoint}")
    # Use Trainer's resume functionality. Pass the checkpoint path (or None) so Trainer can restore trainer state.
    layers = get_layers(model)
    n_layers = len(layers)
    print(f"params={model.num_parameters():,}")
    print(f"layers={n_layers}")

    print(f"gradient_checkpointing={model.is_gradient_checkpointing}")
    print(">>> Acceleration optimizations enabled:")
    print(f"    - Flash Attention: {attn_implementation}")
    print(f"    - Gradient Checkpointing: True")
    print(f"    - Dataloader Workers: 8")
    print(f"    - Memory Pinning: True")
    print(f"    - BF16 Mixed Precision: True")
    print(f"    - Model Compilation: enabled (if PyTorch 2.0+)")
    print(f"    - flash_sdp: {torch.backends.cuda.flash_sdp_enabled()}")

    trainer.train(resume_from_checkpoint=checkpoint)
    return trainer.model

