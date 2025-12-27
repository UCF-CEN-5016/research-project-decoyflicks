import torch
from vector_quantize_pytorch import ResidualVQ

# Set up minimal environment
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_vq = ResidualVQ(num_embeddings=256, embedding_dim=128,
                                       num_residual_layers=2, dim=128)

    def forward(self, x):
        return self.residual_vq(x)

net = Net()

# Add triggering conditions
input_data = torch.randn(1, 3, 32, 32)  # Input data with shape (B, C, H, W)
net.eval()  # Set the model to evaluation mode

# Wrap final code in `python`