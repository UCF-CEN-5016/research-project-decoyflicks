import torch
import numpy as np
from dschat.rlhf.ppo_trainer import DeepSpeedPPOTrainer
from dschat.rlhf.rlhf_engine import DeepSpeedRLHFEngine

torch.manual_seed(42)

batch_size = 16
seq_length = 256

input_prompts = torch.randn(batch_size, seq_length)
action_mask = torch.randint(0, 2, (batch_size, seq_length))
actor_log_prob = torch.randn(batch_size, seq_length)
advantages = torch.randn(batch_size, seq_length)
ground_rewards = torch.randn(batch_size, seq_length)

start = 1
actor_loss_fn = torch.nn.MSELoss()

actor_loss = actor_loss_fn(actor_log_prob[:, start:], ground_rewards[:, start:])

mask = action_mask[:, start-1:-1]
actor_loss_modified = actor_loss_fn(actor_log_prob[:, start:], ground_rewards[:, start:])

assert actor_loss != actor_loss_modified

print(f"Original Actor Loss: {actor_loss.item()}")
print(f"Modified Actor Loss: {actor_loss_modified.item()}")