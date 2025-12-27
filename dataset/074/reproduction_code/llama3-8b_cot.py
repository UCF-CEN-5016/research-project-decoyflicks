import torch
import numpy as np

# Minimal environment setup
batch_size = 2
seq_len = 10
start_token = 0
end_token = 1
action_mask = torch.tensor([[0, 1] + [0]*8 + [1]]).repeat(batch_size, 1)
log_probs = torch.randn((batch_size, seq_len))
actor_log_prob = log_probs.clone()
advantages = torch.randn((batch_size, seq_len))

# Triggering conditions: calculate actor loss
start = start_token
actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])