import torch
import torch.nn.functional as F

# Simulate batch size and sequence length
batch_size = 2
seq_len = 5
start = 2

# Simulated actor_log_prob: log probabilities per token
# For simplicity, random values
actor_log_prob = torch.randn(batch_size, seq_len)
log_probs = torch.randn(batch_size, seq_len)
advantages = torch.randn(batch_size, seq_len)

# Define action_mask: let's say only token at start-1 is True (eos token)
action_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
action_mask[:, start-1] = True  # only eos token is True

# Define a dummy actor_loss_fn that sums up masked losses for illustration
def actor_loss_fn(actor_log_prob_slice, log_probs_slice, advantages_slice, mask):
    # For simplicity, pretend loss = sum of (advantage * log_prob) over masked tokens
    masked_adv = advantages_slice[mask]
    masked_logp = actor_log_prob_slice[mask]
    loss = -(masked_adv * masked_logp).sum()
    return loss

# Current implementation: using mask[:, start:]
loss_current = actor_loss_fn(
    actor_log_prob[:, start:], 
    log_probs[:, start:], 
    advantages[:, start:], 
    action_mask[:, start:]
)

# Suggested mask: mask[:, start-1:-1]
loss_suggested = actor_loss_fn(
    actor_log_prob[:, start:], 
    log_probs[:, start:], 
    advantages[:, start:], 
    action_mask[:, start-1:-1]
)

print("Action mask:\n", action_mask)
print("Mask[:, start:]:\n", action_mask[:, start:])
print("Mask[:, start-1:-1]:\n", action_mask[:, start-1:-1])
print("Loss with mask[:, start:]:", loss_current.item())
print("Loss with mask[:, start-1:-1]:", loss_suggested.item())