import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc1(x))

model = SimpleModel()
input_tensor = torch.randn(1, 10)
timestep = torch.tensor([0])

def run_test():
    if True:  # replace with your condition
        with torch.inference_mode():
            self_cond_output = model(
                model.predictions(input_tensor.clone().detach(), timestep).pred_x_start
            ).detach_()

if __name__ == "__main__":
    run_test()