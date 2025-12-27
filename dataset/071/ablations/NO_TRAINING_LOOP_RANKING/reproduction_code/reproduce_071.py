import argparse
import deepspeed
import torch
import torchvision

def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")
    parser.add_argument("--actor-model", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--reward-model", default="facebook/opt-350m", type=str)
    parser.add_argument("--deployment-type", default="single_gpu", type=str)
    args = parser.parse_args()
    return args

def main(args):
    deepspeed.init_distributed()
    print(args)
    import transformers.deepspeed  # This will raise ModuleNotFoundError if not found

if __name__ == "__main__":
    args = add_argument()
    main(args)