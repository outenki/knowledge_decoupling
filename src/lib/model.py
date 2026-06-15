from typing import Any
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_from_pretrained(model_path: str, attn_implementation: str) -> Any:
    if attn_implementation:
        print(f"- speed up with {attn_implementation}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation=attn_implementation
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    return model


def load_model_from_config_random(config_name: str, attn_implementation: str) -> Any:
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

def freeze_parameters(model, n_layers):
    # freeze all parameters except the last n layers
    for param in model.parameters():
        param.requires_grad = False
    if n_layers == 0:
        return model
    if n_layers > 0:
        unfreeze_layers = get_layers(model)[-n_layers:]
    else:
        unfreeze_layers = get_layers(model)[: -n_layers]
    for layer in unfreeze_layers:
        for param in layer.parameters():
            param.requires_grad = True
    return model

# def unfreeze_parameters(layer):
#     for param in layer.parameters():
#         param.requires_grad = True

def reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()