import argparse
import os
import random
import time
import warnings
import deepspeed
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='DeepSpeed Training Example')
    parser.add_argument('--data', default='dataset/my-gpt2_text_document', type=str)
    parser.add_argument('--world-size', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--gpu', default=None, type=int)
    args = parser.parse_args()

    if args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    model = nn.Linear(10, 10).cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_dataset = datasets.FakeData(size=1000, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        for i, (images, target) in enumerate(train_loader):
            images = images.cuda(args.local_rank)
            target = target.cuda(args.local_rank)

            output = model(images)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.distributed:
                dist.all_reduce(loss)

if __name__ == '__main__':
    main()