import torch
from transformers import RotaryEmbeddings

# Define a sample model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rotary_emb = RotaryEmbeddings(32)

    def forward(self, Q, K):
        Q, K = self.rotary_emb.rotate_queries_and_keys(Q, K)
        return Q, K

model = MyModel()

# Initialize some sample inputs
Q = torch.randn(1, 10, 32)
K = torch.randn(1, 10, 32)

# Run the model with `with xpos`
with torch.no_grad():
    Q, K = model(Q, K)

# Verify that NaNs have propagated
print(f"Q: {torch.isnan(Q).any()}")
print(f"K: {torch.isnan(K).any()}")