import torch
import deepspeed

# Minimal model that triggers the error
class FaultyModel(torch.nn.Module):
    def __init__(self):
        super(FaultyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        output = self.linear(x)
        return output

# Initialize distributed environment correctly
def main():
    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Instantiate the model
    model = FaultyModel().cuda()

    # Sample data
    x = torch.randn(32, 10).cuda()
    
    # Forward pass
    output = model(x)
    print(output)

if __name__ == "__main__":
    main()