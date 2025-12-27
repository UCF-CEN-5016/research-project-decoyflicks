import argparse
import os
import torch

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-a', '--arch', default='resnet18')
    parser.add_argument('--dummy', action='store_true')
    args = parser.parse_args()
    main_worker(args.gpu, 1, args)

def main_worker(gpu, ngpus_per_node, args):
    print("Starting main worker...")
    if torch.backends.mps.is_available():
        print("MPS is available.")
    else:
        print("MPS is not available.")

if __name__ == "__main__":
    main()