import torch
import torch.nn as nn
from labml_helpers.device import DeviceInfo
from torch.optim import Adam as TorchAdam
from labml import monit
from labml_nn.optimizers.adam import Adam as MyAdam
from labml_nn.optimizers.mnist_experiment import Model
from labml_nn.transformers.rope import RotaryPositionalEmbeddings  # Importing the missing class

def test():
    device_info = DeviceInfo(use_cuda=True, cuda_device=0)
    inp = torch.randn((64, 1, 28, 28), device=device_info.device)
    target = torch.ones(64, dtype=torch.long, device=device_info.device)
    loss_func = nn.CrossEntropyLoss()
    model = Model().to(device_info.device)
    
    # Simulating the bug
    rotary_pe = RotaryPositionalEmbeddings(3)  # Incorrect parameter, preserving the bug

    my_adam = MyAdam(model.parameters())
    torch_adam = TorchAdam(model.parameters())
    loss = loss_func(model(inp), target)
    loss.backward()
    
    with monit.section('MyAdam warmup'):
        for i in range(100):
            my_adam.step()
    with monit.section('MyAdam'):
        for i in range(1000):
            my_adam.step()
    with monit.section('TorchAdam warmup'):
        for i in range(100):
            torch_adam.step()
    with monit.section('TorchAdam'):
        for i in range(1000):
            torch_adam.step()

if __name__ == '__main__':
    test()