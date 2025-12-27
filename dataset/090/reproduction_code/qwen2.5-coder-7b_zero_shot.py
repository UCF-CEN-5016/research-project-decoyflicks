import torch
import torch.nn as nn
import torch.distributed as dist


class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        self.to_logits = nn.Linear(5, 10)

    def forward(self, x):
        return {}


class ModelWrapper(nn.Module):
    def __init__(self):
        super(ModelWrapper, self).__init__()
        self.encoder = SimpleEncoder()

    def forward(self, x):
        return self.encoder.forward(x)


def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")


def main():
    setup_distributed()
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank_local = 0

    model_net = ModelWrapper().to(compute_device)
    optimizer = torch.optim.SGD(model_net.parameters(), lr=0.01)

    input_payload = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    for step in range(10):
        optimizer.zero_grad()
        output = model_net(input_payload)
        loss = sum(output.values())
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()