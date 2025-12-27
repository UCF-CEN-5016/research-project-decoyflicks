import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorLoss(nn.Module):
    def forward(self, actor_log_prob, log_probs, advantages, action_mask):
        ratio = torch.exp(actor_log_prob - log_probs)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        masked_loss = loss * action_mask
        return masked_loss.mean()

batch_size = 2
seq_len = 5
start = 1

actor_log_prob = torch.randn(batch_size, seq_len)
log_probs = torch.randn(batch_size, seq_len)
advantages = torch.randn(batch_size, seq_len)
action_mask = torch.ones(batch_size, seq_len)
action_mask[:, -1] = 0  

loss_fn = ActorLoss()
loss1 = loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages[:, start:], action_mask[:, start:])
loss2 = loss_fn(actor_log_prob[:, start-1:-1], log_probs[:, start-1:-1], advantages[:, start-1:-1], action_mask[:, start-1:-1])

print("Original loss:", loss1.item())
print("Modified loss:", loss2.item())