import torch
from vector_quantize_pytorch import ResidualLFQ

# Minimal setup: Define a simple model using ResidualLFQ
class SimpleModel(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_loss_weight):
        super(SimpleModel, self).__init__()
        self.lfq = ResidualLFQ(num_embeddings, embedding_dim, commitment_loss_weight=commitment_loss_weight)

    def forward(self, x, mask=None):
        return self.lfq(x, mask=mask)

# Triggering conditions
num_embeddings = 10
embedding_dim = 14
commitment_loss_weight = 1.0
input_shape = (2, 1851, 1, 14)  # Example input shape
mask_shape = (2, 1851, 1, 1)  # Example mask shape

# Initialize model, input, and mask
model = SimpleModel(num_embeddings, embedding_dim, commitment_loss_weight)
input_data = torch.randn(input_shape)
mask = torch.randn(mask_shape)

# Run the model with mask to trigger the bug
try:
    output = model(input_data, mask=mask)
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("Model ran without raising an exception.")