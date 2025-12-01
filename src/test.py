import torch
from transformers import AutoModelForCausalLM
from transformers import GPT2Tokenizer

from lib.utils import get_device



def get_max_block_size(model):
    block_size = getattr(model.config, "n_positions", None)
    if not block_size:
        block_size = getattr(model.config, "max_position_embeddings", None)
    return block_size

def generate_answer(model, tokenizer, prompt, max_new_tokens=50):
    """
    Evaluate a causal LM on a one sample.
    Automatically truncates prompt if longer than model context.
    """
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    # truncate if too long
    block_size = get_max_block_size(model)
    if block_size:
        max_input_len = block_size - max_new_tokens
    else:
        max_input_len = None
    if block_size is not None and max_input_len is not None and input_ids.size(1) > max_input_len:
        input_ids = input_ids[:, -max_input_len:]
        attention_mask = attention_mask[:, -max_input_len:]

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            use_cache=True
        )

    gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return gen_text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', dest='model', type=str, default="gpt2")
    parser.add_argument('--prompt', '-p', dest='prompt', type=str, default="Climate is generally described in terms of what?")
    args = parser.parse_args()

    # ======== Set device ========
    device = get_device()
    print(f"Using device: {device}")

    # ======== Load model and tokenizer ========
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_name = args.model
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    prompt = ""
    # prompt = "Answer the question briefly in one sentence.\n"
    prompt += args.prompt.strip()
    pred = generate_answer(model, tokenizer, prompt).strip()
    print(">>> prompt:\n", prompt)
    print(">>> pred:\n", pred)


main()