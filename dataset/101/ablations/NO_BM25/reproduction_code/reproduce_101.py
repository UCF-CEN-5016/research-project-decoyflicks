import torch
from rotary_embedding_torch import RotaryEmbedding

# Set device for computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dim = 64
rotary_emb = RotaryEmbedding(dim=dim, use_xpos=True)

# Initialize input data
batch_size = 8
seq_len = 16
Q = torch.randn(batch_size, seq_len, dim).to(device)
K = torch.randn(batch_size, seq_len, dim).to(device)

# Rotate queries and keys
rotated_Q, rotated_K = rotary_emb.rotate_queries_and_keys(Q, K)

# Check for NaNs in rotated_K
assert not torch.isnan(rotated_K).any(), 'rotated_K contains NaN values'
print(rotated_K[torch.isnan(rotated_K)])  # This will print NaNs if they exist

# Placeholder for transformer model and loss function
# These should be defined elsewhere in the actual implementation
# For the purpose of reproducing the bug, we will assume they are defined
# transformer_model = ...
# loss_function = ...
# target = ...

# Perform transformer operations (commented out to avoid undefined variable errors)
# output = transformer_model(rotated_Q, rotated_K)
# loss = loss_function(output, target)

# Check for NaNs in loss (commented out to avoid undefined variable errors)
# assert not torch.isnan(loss).any(), 'Loss contains NaN values'
# print(loss)