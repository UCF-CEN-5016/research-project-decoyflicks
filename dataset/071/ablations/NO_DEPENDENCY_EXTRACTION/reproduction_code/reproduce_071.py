import argparse
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")
    parser.add_argument("--actor-model", type=str, required=True)
    parser.add_argument("--reward-model", type=str, required=True)
    parser.add_argument("--deployment-type", type=str, required=True)
    parser.add_argument("-e", "--epochs", default=30, type=int, help="number of total epochs (default: 30)")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser.add_argument("--log-interval", type=int, default=2000, help="output logging information at a given interval")
    parser.add_argument("--dtype", default="fp16", type=str, choices=["bf16", "fp16", "fp32"], help="Datatype used for training")
    parser.add_argument("--stage", default=0, type=int, choices=[0, 1, 2, 3], help="Datatype used for training")
    parser.add_argument("--moe", default=False, action="store_true", help="use deepspeed mixture of experts (moe)")
    parser.add_argument("--ep-world-size", default=1, type=int, help="(moe) expert parallel world size")
    parser.add_argument("--num-experts", type=int, nargs="+", default=[1], help="number of experts list, MoE related.")
    parser.add_argument("--mlp-type", type=str, default="standard", help="Only applicable when num-experts > 1, accepts [standard, residual]")
    parser.add_argument("--top-k", default=1, type=int, help="(moe) gating top 1 and 2 supported")
    parser.add_argument("--min-capacity", default=0, type=int, help="(moe) minimum capacity of an expert regardless of the capacity_factor")
    parser.add_argument("--noisy-gate-policy", default=None, type=str, help="(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter")
    parser.add_argument("--moe-param-group", default=False, action="store_true", help="(moe) create separate moe param groups, required when using ZeRO w. MoE")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.moe = args.moe
        if self.moe:
            fc3 = nn.Linear(84, 84)
            self.moe_layer_list = []
            for n_e in args.num_experts:
                self.moe_layer_list.append(deepspeed.moe.layer.MoE(hidden_size=84, expert=fc3, num_experts=n_e, ep_size=args.ep_world_size, use_residual=args.mlp_type == "residual", k=args.top_k, min_capacity=args.min_capacity, noisy_gate_policy=args.noisy_gate_policy))
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
            self.fc4 = nn.Linear(84, 10)
        else:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.moe:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x

def main(args):
    deepspeed.init_distributed()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    net = Net(args)
    parameters = filter(lambda p: p.requires_grad, net.parameters())
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
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, model=net, model_parameters=parameters, training_data=trainset, config=ds_config)

if __name__ == "__main__":
    args = add_argument()
    main(args)