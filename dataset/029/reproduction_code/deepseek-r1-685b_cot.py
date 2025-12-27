import torch
import platform

# This will fail on non-macOS systems
try:
    if torch.backends.mps.is_available():
        print("MPS available")
except AttributeError:
    print("Error: torch.backends.mps not available (expected on non-macOS systems)")
    print(f"System: {platform.system()}")
    print(f"PyTorch version: {torch.__version__}")

# Expected output on Linux/Windows:
# Error: torch.backends.mps not available (expected on non-macOS systems)
# System: Linux
# PyTorch version: x.x.x