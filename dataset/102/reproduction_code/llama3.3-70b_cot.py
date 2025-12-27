import torch
import torch.nn as nn

class ResidualSimVQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, quantize_dropout=0.0):
        super(ResidualSimVQ, self).__init__()
        self.quantize_dropout = quantize_dropout
        self.training = True  # Set to True to trigger the bug

    def forward(self, x):
        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss  # This line triggers the bug
        # Simulate the quantization process
        quantized_x = x
        # Simulate the shape inconsistency issue
        if self.quantize_dropout:
            quantized_x = quantized_x.permute(0, 2, 1)  # channels_first
        return quantized_x

# Set up the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the ResidualSimVQ class
residual_sim_vq = ResidualSimVQ(num_embeddings=1024, embedding_dim=1024, quantize_dropout=0.1)

# Move the instance to the device
residual_sim_vq.to(device)

# Create a random input tensor
input_tensor = torch.randn(2, 17, 1024).to(device)

# Trigger the bug
try:
    output = residual_sim_vq(input_tensor)
except NameError as e:
    print(e)