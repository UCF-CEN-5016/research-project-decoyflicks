import torch

class ValuePE:
    def __init__(self):
        pass

    def rotate(self, x):
        rotation_matrix = torch.tensor([[0, -1], [1, 0]], device=x.device)
        return torch.matmul(x, rotation_matrix)

    def forward(self, x):
        x = self.rotate(x)
        x = self.rotate(x)
        return x

x = torch.tensor([[1., 2.], [3., 4.]])
value_pe = ValuePE()
result = value_pe.forward(x)
print(result)