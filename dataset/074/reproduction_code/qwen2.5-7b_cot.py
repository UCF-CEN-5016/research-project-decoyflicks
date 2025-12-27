import torch

# Set up dummy data
batch_size = 1
seq_len = 5
start = 5  # This is the index that causes the EOS token to be excluded

# Create a dummy action_mask where the EOS token is at index 4 (masked as 1)
# and all other tokens are also valid (masked as 1)
action_mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
action_mask[:, start-1] = 0

# Original mask: action_mask[:, start:] (excludes the EOS token)
original_mask = action_mask[:, start:]
print("Original mask (excludes EOS token):", original_mask)

# Suggested mask: action_mask[:, start-1:] (includes the EOS token)
suggested_mask = action_mask[:, start-1:]
print("Suggested mask (includes EOS token):", suggested_mask)

# Check that the original mask is empty
assert original_mask.size(1) == 0, "Original mask should exclude the EOS token."

# Check that the suggested mask includes the EOS token
assert suggested_mask.size(1) == 1, "Suggested mask should include the EOS token."

action_mask[:, start-1:]