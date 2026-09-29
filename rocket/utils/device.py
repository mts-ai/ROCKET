import os
import torch

# Linear algebra runs in float64, which MPS does not support, so on Apple
# silicon it falls back to CPU. Override with ROCKET_LINALG_DEVICE.
LINALG_DEVICE = os.environ.get("ROCKET_LINALG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def model_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
