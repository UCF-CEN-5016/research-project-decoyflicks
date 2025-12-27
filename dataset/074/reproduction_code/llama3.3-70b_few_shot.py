import torch
import torch.nn.functional as F

# Sample data
batch_size = 32
sequence_length = 10
action_dim = 5

actor_log_prob = torch.randn(batch_size, sequence_length, action_dim)
log_probs = torch.randn(batch_size, sequence_length, action_dim)
advantages = torch.randn(batch_size, sequence_length, action_dim)
action_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)

# Incorrect masking
start = 1
actor_loss_fn = lambda log_prob, log_probs, advantages, mask: torch.sum(log_prob * advantages * mask) / torch.sum(mask)
incorrect_actor_loss = actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages[:, start:], action_mask[:, start:])

# Correct masking
correct_actor_loss = actor_loss_fn(actor_log_prob[:, start-1:-1], log_probs[:, start-1:-1], advantages[:, start-1:-1], action_mask[:, start-1:-1])

# Print the results
print("Incorrect Actor Loss:", incorrect_actor_loss)
print("Correct Actor Loss:", correct_actor_loss)