import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18", type=str, help="model architecture")
    parser.add_argument("--gpu", default=0, type=int, help="GPU ID")
    args = parser.parse_args()
    
    print(f"Using model architecture: {args.arch} on GPU ID: {args.gpu}")
    
    if args.gpu == 0:
        print("Using the first GPU")
    
    try:
        if not torch.backends.mps.is_available():
            print("MPS backend is not available.")
    except AttributeError as e:
        print(f"AttributeError: {e}")

if __name__ == "__main__":
    main()