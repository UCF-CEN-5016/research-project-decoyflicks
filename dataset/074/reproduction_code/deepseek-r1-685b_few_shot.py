import torch
import numpy as np

# Simulate RL training setup with sequence generation
seq_len = 10
batch_size = 4
vocab_size = 100
start_token = 1  # Special start token
eos_token = 2    # Special EOS token

# Simulate model outputs
actor_log_prob = torch.randn(batch_size, seq_len, vocab_size)
log_probs = torch.randn(batch_size, seq_len, vocab_size)
advantages = torch.randn(batch_size, seq_len)

# Create action mask (1=valid, 0=invalid)
action_mask = torch.ones(batch_size, seq_len)
for i in range(batch_size):
    eos_pos = np.random.randint(3, seq_len-1)  # Random EOS position
    action_mask[i, eos_pos+1:] = 0  # Mask tokens after EOS

# Current (buggy) implementation
start = 1  # Skip start token
buggy_loss = (actor_log_prob[:, start:] * 
             log_probs[:, start:] * 
             advantages.unsqueeze(-1) * 
             action_mask[:, start:].unsqueeze(-1))

# Proposed fix implementation
fixed_loss = (actor_log_prob[:, start-1:-1] * 
             log_probs[:, start-1:-1] * 
             advantages.unsqueeze(-1)[:, start-1:-1] * 
             action_mask[:, start-1:-1].unsqueeze(-1))

# Compare results
print("Original mask shape:", action_mask[:, start:].shape)
print("Original loss includes EOS:", buggy_loss[:, eos_pos-start].sum() != 0)
print("Fixed mask shape:", action_mask[:, start-1:-1].shape)
print("Fixed loss includes EOS:", fixed_loss[:, eos_pos-start+1].sum() != 0)