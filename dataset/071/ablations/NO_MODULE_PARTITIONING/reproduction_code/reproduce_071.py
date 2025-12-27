import argparse
import deepspeed
import torch
import torchvision
from deepspeed.accelerator import get_accelerator

def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")
    parser.add_argument("--actor-model", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--reward-model", default="facebook/opt-350m", type=str)
    parser.add_argument("--deployment-type", default="single_gpu", type=str)
    args = parser.parse_args()
    return args

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main(args):
    deepspeed.init_distributed()
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    net = Net(args)
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, model=net, model_parameters=net.parameters(), training_data=trainset)
    local_device = get_accelerator().device_name(model_engine.local_rank)
    for epoch in range(1):  # Just for testing
        for data in trainloader:
            inputs, labels = data[0].to(local_device), data[1].to(local_device)
            outputs = model_engine(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

if __name__ == "__main__":
    args = add_argument()
    main(args)