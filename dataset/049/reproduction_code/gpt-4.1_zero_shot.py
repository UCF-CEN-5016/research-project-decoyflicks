import torch
class RelPosBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.u = torch.Tensor(2, 2)
        self.v = torch.Tensor(2, 2)
    def forward(self):
        return self.u.sum() + self.v.sum()

m = RelPosBias()
print(m.u)
print(m.v)
print(m())