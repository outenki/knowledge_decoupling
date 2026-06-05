from typing import Any
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_from_pretrained(model_path: str, attn_implementation: str) -> Any:
    print(">>> Loading init model from:", model_path)
    if attn_implementation:
        print(f"- speed up with {attn_implementation}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation=attn_implementation
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    print(">>> Init model loaded. Config:", model.config)
    return model


def load_model_from_config_random(config_name: str, attn_implementation: str) -> Any:
    print(">>> Initializing model from random weights using config:", config_name)
    config = AutoConfig.from_pretrained(config_name, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config
    if attn_implementation:
        print(f"- speed up with {attn_implementation}")
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True
        )
    for name, module in model.named_modules():
        if hasattr(module, "use_linear_attn"):
            module.use_linear_attn = False
    model.tie_weights()
    # Enable gradient checkpointing to speed up training (trade memory for speed)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print(">>> Checking random model parameters...")
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() > 1:
            print(name, p.mean().item(), p.std().item())
            break
    return model.to(torch.bfloat16)
