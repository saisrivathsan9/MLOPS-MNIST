import torch

def get_device():
    """
    Returns GPU device if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")