import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple M1/M2
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
