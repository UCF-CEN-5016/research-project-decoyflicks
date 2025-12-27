import torch
import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine

# Minimal model that triggers the error
class FaultyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        # This will trigger the error because comm is not initialized
        output = self.linear(x)
        deepspeed.comm.all_reduce(output)  # Error occurs here
        return output

# Initialize distributed environment incorrectly
def main():
    # Missing proper DeepSpeed initialization
    model = FaultyModel()
    engine = DeepSpeedEngine(model, config={})
    
    # Sample data
    x = torch.randn(32, 10).cuda()
    
    # Forward pass triggers error
    output = engine(x)
    print(output)

if __name__ == "__main__":
    main()