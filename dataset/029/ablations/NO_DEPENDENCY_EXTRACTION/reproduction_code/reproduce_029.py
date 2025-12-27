import torch
import torch.nn as nn
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    
    model = nn.Linear(10, 5)
    
    if torch.backends.mps.is_available():
        print('MPS is available')
    else:
        print('MPS is not available')

if __name__ == "__main__":
    main()