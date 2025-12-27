import torch

# Create fake data
actor_log_prob = torch.randn(1, 5)
log_probs = torch.randn(1, 5)
advantages = torch.randn(1, 5)
action_mask = torch.tensor([[1, 1, 1, 1, 0]])  # EOS at position 4, mask is 0

start = 4

# Original code's mask (excludes EOS)
original_mask = action_mask[:, start:]  # [0]
# User's suggested mask (includes EOS)
user_mask = action_mask[:, start-1:-1]  # [1, 1, 1]

# Simulate loss calculation (dummy)
def compute_loss(log_probs, advantages, mask):
    return (log_probs * advantages * mask).sum()

original_loss = compute_loss(actor_log_prob, advantages, original_mask)
user_loss = compute_loss(log_probs, advantages, user_mask)

print("Original loss:", original_loss.item())
print("User's loss:", user_loss.item())