import torch

def check_mps_availability():
    if torch.backends.mps.is_available():
        return "MPS is available"
    else:
        return "MPS is not available"

print(check_mps_availability())