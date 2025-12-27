import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def main():
    model = SimpleModel(input_dim=10, num_classes=50)
    input_data = torch.randn(32, 10)
    log_probs = model(input_data)

    if log_probs.dim() != 3:
        raise RuntimeError("log_probs must be 3-D (batch_size, input length, num classes)")

if __name__ == "__main__":
    main()