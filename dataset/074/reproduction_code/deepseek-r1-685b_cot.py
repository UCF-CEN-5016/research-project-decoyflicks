import torch
import torch.nn as nn

# Mock components
class ActorModel(nn.Module):
    def forward(self, x):
        # Returns log_probs for 3 tokens (batch_size=1, seq_len=3, vocab_size=5)
        return torch.randn(1, 3, 5).log_softmax(dim=-1)

# Configuration
batch_size = 1
seq_len = 3
vocab_size = 5
start = 1  # Typically where we start calculating loss after prompt

# Mock data
actor_log_prob = ActorModel()(None)  # shape (1, 3, 5)
log_probs = torch.randn(1, 3, 5).log_softmax(dim=-1)
advantages = torch.tensor([[0.5, 1.0, -0.3]])  # Includes reward for eos
action_mask = torch.tensor([[0, 1, 1]])  # Mask (0 for prompt, 1 for generated tokens)

# Original buggy implementation
def actor_loss_fn(logp, old_logp, adv, mask):
    loss = -adv * (logp - old_logp) * mask
    return loss.mean()

# Original (buggy) call
buggy_loss = actor_loss_fn(
    actor_log_prob[:, start:],
    log_probs[:, start:],
    advantages[:, start:],
    action_mask[:, start:]
)

# Proposed fix
fixed_loss = actor_loss_fn(
    actor_log_prob[:, start-1:],  # Include one more position
    log_probs[:, start-1:],
    advantages[:, start-1:],
    action_mask[:, start-1:-1]  # Adjusted mask
)

print("Buggy loss (excludes eos reward):", buggy_loss)
print("Fixed loss (includes eos reward):", fixed_loss)

# Explanation
print("\nKey difference:")
print("Original mask:", action_mask[:, start:].tolist(), "-> misses eos reward")
print("Proposed mask:", action_mask[:, start-1:-1].tolist(), "-> includes eos reward")
print("Original advantages:", advantages[:, start:].tolist(), "vs full:", advantages.tolist())