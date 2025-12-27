import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pad = nn.ConstantPad1d((0, 2), 0)

    def align_right(self, x):
        return self.pad(x.view(-1, 1)).view(-1)

model = Model()
print(model.align_right(torch.tensor([1, 2, 3, 4, 5])))