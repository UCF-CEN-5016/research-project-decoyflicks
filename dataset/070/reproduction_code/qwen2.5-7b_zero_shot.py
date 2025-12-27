import torch
import deepspeed

def setup_model_optimizer():
    model = torch.nn.Linear(10, 10).to('cuda')
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer

def train_step(model, optimizer):
    data = torch.randn(10, 10).to('cuda')
    loss = model(data).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

if __name__ == "__main__":
    model, optimizer = setup_model_optimizer()
    with deepspeed.zero.Init():
        train_step(model, optimizer)