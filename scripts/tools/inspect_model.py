
import argparse
import torch

from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights


def inspect_model(model_path: str):
    print("=" * 70)
    print(f"Model path: {model_path}")
    print("=" * 70)

    # 1. Load model config
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if hasattr(config, "text_config"):
        config = config.text_config

    # 2. Print architecture information
    attributes = [
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "tie_word_embeddings",
    ]

    print("\n[Architecture]")
    for attr in attributes:
        if hasattr(config, attr):
            print(f"{attr:30s}: {getattr(config, attr)}")

    # 3. Build model on Meta Device
    # No actual model weights are allocated.
    print("\n[Building model on Meta Device]")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )

    # Handle tied weights
    model.tie_weights()

    # 4. Count parameters
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n[Parameter Count]")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params / 1e6:.2f}M")
    print(f"Total parameters:     {total_params / 1e9:.4f}B")

    # 5. Count parameters by component
    print("\n[Parameter Breakdown]")

    components = {
        "Embedding": 0,
        "Attention": 0,
        "MLP / FFN": 0,
        "Norm": 0,
        "LM Head": 0,
        "Other": 0,
    }

    for name, param in model.named_parameters():
        n = name.lower()
        count = param.numel()

        if "embed_tokens" in n or "wte" in n:
            components["Embedding"] += count
        elif "lm_head" in n:
            components["LM Head"] += count
        elif "self_attn" in n or "attention" in n:
            components["Attention"] += count
        elif "mlp" in n or "feed_forward" in n:
            components["MLP / FFN"] += count
        elif "norm" in n:
            components["Norm"] += count
        else:
            components["Other"] += count

    for name, count in components.items():
        print(
            f"{name:20s}: "
            f"{count:15,} ({count / 1e6:10.3f}M)"
        )

    print("-" * 70)

    # 6. Verify embedding / LM head sharing
    if hasattr(model, "get_output_embeddings"):
        input_emb = model.get_input_embeddings()
        output_emb = model.get_output_embeddings()

        if input_emb is not None and output_emb is not None:
            shared = (
                input_emb.weight is output_emb.weight
            )
            print(f"Embedding shared: {shared}")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        type=str,
        help="Local Hugging Face model directory",
    )

    args = parser.parse_args()

    inspect_model(args.model_path)
