import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set random seed for reproducibility
torch.manual_seed(0)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)

def model_predictions(model, x):
    # Simulate model prediction output object with pred_x_start attribute
    class Output:
        def __init__(self, pred):
            self.pred_x_start = pred
    pred = model(x)
    return Output(pred)

def main():
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(2, 4, requires_grad=True)
    t = torch.tensor([1, 2])  # dummy timestep

    self_condition = True

    for step in range(5):
        optimizer.zero_grad()

        if self_condition and random.random() < 0.5:
            # Using torch.inference_mode() - this will cause the error when backward is called
            with torch.inference_mode():
                # Forward pass in inference mode
                x_self_cond = model_predictions(model, x.clone().detach()).pred_x_start
                x_self_cond = x_self_cond.detach_()

            # Using x_self_cond in computation graph - this causes the RuntimeError on backward
            loss = (x_self_cond * x).sum()
        else:
            # Normal forward pass with grad tracking
            output = model(x)
            loss = output.sum()

        # Backward pass - triggers error when x_self_cond is inference tensor
        loss.backward()
        optimizer.step()

        print(f"Step {step} completed")

if __name__ == "__main__":
    main()