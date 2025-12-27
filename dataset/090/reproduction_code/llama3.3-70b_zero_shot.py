import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.to_logits = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()

    def forward(self, x):
        x = self.encoder(x)
        return x

def main():
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    model = Model()
    device = torch.device("cuda")
    model.to(device)
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    input = torch.randn(1, 10).to(device)
    for i in range(10):
        optimizer.zero_grad()
        output = model(input)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        print(model.module.encoder.to_logits.weight.grad)

if __name__ == "__main__":
    main()