import torch
from einops import rearrange

# Minimal setup
class RotaryXPos:
    def __init__(self, scale):
        self.scale = scale

    def forward(self, power):
        # Triggering condition: rotary_xpos is turned on
        # and the tensor has a shape that causes the error
        scale = self.scale ** rearrange(power, 'n -> n 1')
        return scale

# Set up minimal environment
if __name__ == "__main__":
    # Define a tensor that triggers the bug
    power = torch.randn(1, 32)  # 2-dimensional tensor
    
    # Create an instance of RotaryXPos with a scale value
    rotary_xpos = RotaryXPos(scale=2.0)
    
    # Call the forward method to trigger the bug
    try:
        rotary_xpos.forward(power)
    except Exception as e:
        print(f"Error: {e}")