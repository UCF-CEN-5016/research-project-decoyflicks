import torch

# Example sequence of length 5
batch_size = 1
sequence_length = 5
action_log_prob = torch.randn(batch_size, sequence_length)
log_probs = torch.randn(batch_size, sequence_length)
advantages = torch.randn(batch_size, sequence_length)
action_mask = torch.tensor([
    [True, True, True, True, False],  # Index 0: action, 1: action, 2: action, 3: action, 4: EOS
], dtype=torch.bool)

start = 3  # Assume the actions start at index 3

# Original mask: includes the last token (EOS), but it's marked as False
original_mask = action_mask[:, start:]
print("Original mask:", original_mask)

# Suggested mask: excludes the last token (EOS)
suggested_mask = action_mask[:, start-1:-1]
print("Suggested mask:", suggested_mask)

# Correct mask (includes all tokens from start-1 to end)
correct_mask = action_mask[:, start-1:]
print("Correct mask (includes EOS):", correct_mask)