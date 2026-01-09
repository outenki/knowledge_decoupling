import torch


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