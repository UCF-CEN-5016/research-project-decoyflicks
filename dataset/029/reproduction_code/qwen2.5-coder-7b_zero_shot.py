import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def create_model() -> SimpleNet:
    return SimpleNet()


def print_mps_status() -> None:
    if torch.backends.mps.is_available():
        print("MPS is available")
    else:
        print("MPS is not available")


if __name__ == "__main__":
    model = create_model()
    print_mps_status()