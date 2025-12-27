import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gpu == 0:
        print(f"Creating model: {args.arch}")

    try:
        mps_available = torch.backends.mps.is_available()
    except AttributeError as e:
        print(f"AttributeError: {e}")

if __name__ == "__main__":
    main()