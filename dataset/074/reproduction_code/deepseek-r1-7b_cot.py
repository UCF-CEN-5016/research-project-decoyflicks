import torch
from transformers import AutoModelForCausalLM

# Create a dummy model and environment
model = AutoModelForCausalLM()

# Simulate token distribution, action_mask, log_probs, and advantages
batch_size = 32
sequence_length = 50
actor_log_prob = torch.randn(batch_size, sequence_length)
log_probs = torch.randn(batch_size, sequence_length)
advantages = torch.randn(batch_size, sequence_length)
action_mask = torch.ones(batch_size, sequence_length)  # Assuming all tokens are valid actions

# Apply the suggested change to include 'eos'
start = 5  # Example start index
mask = action_mask[:, start-1:-1] if start > 0 else action_mask[:, :]

actor_loss = model.actor_loss_fn(
    actor_log_prob[:, start:],
    log_probs[:, start:], 
    advantages,
    mask
)

print("Adjusted Mask Shape:", mask.shape)