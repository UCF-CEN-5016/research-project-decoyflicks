import torch

def check_mps_availability():
    try:
        if torch.backends.mps.is_available():
            print("MPS backend is available.")
        else:
            print("MPS backend is not available.")
    except AttributeError as e:
        print(f"AttributeError: {e}")

if __name__ == "__main__":
    check_mps_availability()