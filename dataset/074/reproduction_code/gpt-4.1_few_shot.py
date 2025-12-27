import torch
import torch.nn.functional as F

# Simulate logits and log_probs for a sequence batch
batch_size = 2
seq_len = 5
vocab_size = 10

# Random log probabilities for tokens at each position
log_probs = torch.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)

# Suppose actions taken (indices) and corresponding log_probs gathered
actions = torch.randint(0, vocab_size, (batch_size, seq_len))
actor_log_prob = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

# Advantages (rewards) per token
advantages = torch.randn(batch_size, seq_len)

# Action mask indicating which tokens are valid for loss calculation
# Mark last token (eos) as valid (1)
action_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
# For demonstration, assume start index after some prefix tokens
start = 2

# Incorrect mask slicing excludes eos token at position seq_len-1
incorrect_mask = action_mask[:, start:]  

# Correct mask slicing includes eos token aligned with shifted tokens
correct_mask = action_mask[:, start-1:-1]  

# Define a simple actor loss function (negative log likelihood weighted by advantages)
def actor_loss_fn(actor_log_prob_slice, log_probs_slice, advantages_slice, mask):
    # Mask out invalid positions
    masked_log_prob = actor_log_prob_slice[mask]
    masked_adv = advantages_slice[mask]
    # Negative weighted log likelihood
    return -(masked_log_prob * masked_adv).mean()

# Calculate loss with incorrect mask
loss_incorrect = actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages, incorrect_mask)

# Calculate loss with correct mask
loss_correct = actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages, correct_mask)

print(f"Loss with incorrect mask (excludes eos): {loss_incorrect.item():.4f}")
print(f"Loss with correct mask (includes eos): {loss_correct.item():.4f}")