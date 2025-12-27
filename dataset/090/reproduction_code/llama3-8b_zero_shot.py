import torch
import torch.nn as nn
import torch.distributed as dist

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.to_logits = nn.Linear(5, 10)

    def forward(self, x):
        return {}

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = Encoder()

    def forward(self, x):
        return self.encoder.forward(x)

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    model = MyModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(10):
        optimizer.zero_grad()
        output = model({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
        loss = sum(output.values())
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()