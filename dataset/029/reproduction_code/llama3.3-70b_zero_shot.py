import torch

def main():
    if torch.backends.mps.is_available():
        print("MPS is available")
    else:
        print("MPS is not available")

if __name__ == "__main__":
    main()