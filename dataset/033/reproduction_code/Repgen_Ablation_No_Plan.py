import torch

class TestMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(5, 1),
            torch.nn.ReLU(),
            torch.nn.Linear(1, 2),
        )

    def forward(self, inputs):
        return self.layers(inputs)

def main():
    model = TestMod()
    model.cuda()
    model.half()
    model.train()
    x = torch.randn(8910, 3).cuda().half()
    with torch.no_grad():
        out = model(x)
        print(out.shape)  # Expected shape: [8910, 2]

if __name__ == '__main__':
    main()