import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", default="resnet18", type=str)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--gpu", default=0, type=int)
    args = parser.parse_args()

    if args.gpu == 0:
        print("Using GPU 0")

    print(torch.backends.mps.is_available())

if __name__ == "__main__":
    main()