import torch

def _mps_is_available() -> bool:
    """Check whether the MPS backend is available."""
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False

def get_mps_availability_message() -> str:
    """Return a message describing MPS availability."""
    return "MPS is available" if _mps_is_available() else "MPS is not available"

if __name__ == "__main__":
    print(get_mps_availability_message())