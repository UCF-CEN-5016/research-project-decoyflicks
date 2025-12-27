import argparse
from deepspeed import add_config_arguments, convert_to_random_ltd, init_distributed, initialize, save_without_random_ltd
import random
import shutil
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import Subset
from torch.distributed import all_reduce, init_process_group, load, manual_seed, no_grad, save, spawn
from torchvision.transforms import CenterCrop, Compose, FakeData, ImageFolder, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor

def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    add_config_arguments(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    init_process_group(backend=args.backend, world_size=args.world_size, rank=args.rank)

    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    model = ...  # Initialize your model here
    optimizer = ...  # Initialize your optimizer here

    criterion = CrossEntropyLoss()

    train_dataset = FakeData(10, (3, 224, 224), transform=Compose([Resize((256, 256)), RandomCrop(224), ToTensor()]))
    val_dataset = ImageFolder('path_to_val_data', transform=Compose([CenterCrop(224), ToTensor()]))

    train_sampler = ...  # Initialize your sampler here
    val_sampler = ...  # Initialize your sampler here

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    if args.distributed:
        model = convert_to_random_ltd(model)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, criterion, optimizer, train_loader, device)
        val_loss, val_acc = validate(model, criterion, val_loader, device)

        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    save(model.state_dict(), 'model.pth')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        shutil.copyfile('model.pth', 'model_best.pth')

def train(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / len(dataloader.dataset) * 100
    return epoch_loss, epoch_acc

def validate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0

    with no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / len(dataloader.dataset) * 100
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    main()