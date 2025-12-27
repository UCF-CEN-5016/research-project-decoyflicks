import argparse
import deepspeed
import torch
import torchvision
from deepspeed import add_config_arguments

def add_argument():
    parser = argparse.ArgumentParser(description="RLHF")
    parser.add_argument("--actor-model", type=str, required=True)
    parser.add_argument("--reward-model", type=str, required=True)
    parser.add_argument("--deployment-type", type=str, required=True)
    args = parser.parse_args()
    return args

def get_ds_config(args):
    ds_config = {
        "train_batch_size": 16,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 0,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    return ds_config

class Net(torch.nn.Module):
    def __init__(self):
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
    
    net = Net()
    ds_config = get_ds_config(args)
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=net.parameters(),
        training_data=trainset,
        config=ds_config,
    )

    for epoch in range(1):  # loop over the dataset once
        for data in trainloader:
            inputs, labels = data
            outputs = model_engine(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

if __name__ == "__main__":
    args = add_argument()
    main(args)