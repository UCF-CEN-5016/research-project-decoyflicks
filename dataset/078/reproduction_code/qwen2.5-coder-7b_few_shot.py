import torch

def make_nested_input():
    return torch.nested.nested_tensor([
        torch.randn(5, 1024),
        torch.randn(5, 1024)
    ])

class NestedLinearModel(torch.nn.Module):
    def __init__(self, in_features: int = 1024, out_features: int = 1024):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nested.map(self.linear, x)

def main():
    model = NestedLinearModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    nested_tensors = make_nested_input()

    output = model(nested_tensors)
    loss = output.sum()
    loss.backward()

if __name__ == "__main__":
    main()