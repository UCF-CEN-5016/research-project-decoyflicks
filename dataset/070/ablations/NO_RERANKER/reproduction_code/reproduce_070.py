import argparse
import os
import random
import shutil
import time
import warnings
import deepspeed
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR  # Added import for StepLR
import torch.backends.cudnn as cudnn  # Added import for cudnn

# Placeholder functions for undefined functions in the original code
def _get_dist_model(gpu, args):
    # Placeholder for the actual model retrieval logic
    return nn.Linear(10, 10).cuda(gpu)

def _get_model(args):
    # Placeholder for the actual model retrieval logic
    return nn.Linear(10, 10).cuda(args.gpu)

def validate(val_loader, model, criterion, args):
    # Placeholder for the actual validation logic
    return 0, 0, 0  # Dummy return values

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # Placeholder for the actual checkpoint saving logic
    pass

def save_without_random_ltd(model):
    # Placeholder for the actual model saving logic
    return model.state_dict()

class AverageMeter:
    # Placeholder for the AverageMeter class
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ProgressMeter:
    # Placeholder for the ProgressMeter class
    def __init__(self, num_batches, meters, prefix=""):
        self.meters = meters
        self.num_batches = num_batches
        self.prefix = prefix

    def display(self, batch):
        # Placeholder for display logic
        pass

def accuracy(output, target, topk=(1,)):
    # Placeholder for the accuracy calculation logic
    return (torch.tensor([0.0]), torch.tensor([0.0]))  # Dummy return values

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global history
    history = {"epoch": [], "val_loss": [], "val_acc1": [], "val_acc5": [], "train_loss": [], "train_acc1": [], "train_acc5": []}  # Initialize history

    if args.deepspeed:
        gpu = args.local_rank
    args.gpu = gpu
    ngpus = torch.cuda.device_count()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if not args.deepspeed:
        model = _get_dist_model(gpu, args)
    else:
        model = _get_model(args)
        deepspeed.init_distributed()
    print(f'created model on gpu {gpu}')

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,  # Moved optimizer initialization here
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.RandomCrop(args.img_size, padding=(args.img_size//8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    args.completed_step = 0

    scheduler = StepLR(optimizer, step_size=int(len(train_loader)*args.epochs//3), gamma=0.1)

    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        args=args,
        dist_init_required=False)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        start_time = time.time()
        top5_train, top1_train, losses_train, args = train(scheduler, train_loader, model, criterion, optimizer, epoch, args)
        time_epoch = time.time() - start_time
        top5_val, top1_val, losses_val = validate(val_loader, model, criterion, args)

        if args.gpu == 0:
            history["epoch"].append(epoch)
            history["val_loss"].append(losses_val)
            history["val_acc1"].append(top1_val)
            history["val_acc5"].append(top5_val)
            history["train_loss"].append(losses_train)
            history["train_acc1"].append(top1_train)
            history["train_acc5"].append(top5_train)
            torch.save(history, f"{args.out_dir}/stat.pt")
            print(f'{epoch} epoch at time {time_epoch}s and learning rate {scheduler.get_last_lr()}')
            print(f"finish epoch {epoch} or iteration {args.completed_step}, train_accuracy is {top1_train}, val_accuracy {top1_val}")

        is_best = top1_val > best_acc1
        best_acc1 = max(top1_val, best_acc1)
        if is_best:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0) or args.local_rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': save_without_random_ltd(model),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict()
                }, is_best, filename=f"{args.out_dir}/checkpoint.pth.tar")

def train(scheduler, train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    lr_progress = AverageMeter('lr', ':0f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, lr_progress],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        args.completed_step += 1
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.clone().detach(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        lr_progress.update(scheduler.get_last_lr()[0], 1)
        if args.deepspeed:
            model.backward(loss)
            model.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

    if args.distributed:
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    return top5.avg, top1.avg, losses.avg, args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Add arguments as per the original script
    # ...
    args = parser.parse_args()
    main_worker(args.gpu, torch.cuda.device_count(), args)