import torch
import torch.nn.functional as F

def actor_loss_fn(log_prob, old_log_prob, advantages, mask):
    ratio = (log_prob - old_log_prob).exp()
    loss = - (ratio * advantages * mask).sum() / mask.sum()
    return loss

batch_size = 2
seq_len = 5
start = 1

actor_log_prob = torch.log_softmax(torch.randn(batch_size, seq_len), dim=-1)
log_probs = torch.log_softmax(torch.randn(batch_size, seq_len), dim=-1)
advantages = torch.randn(batch_size, seq_len)
action_mask = torch.tensor([[1,1,1,1,1],
                            [1,1,1,1,0]], dtype=torch.float32)

loss1 = actor_loss_fn(actor_log_prob[:, start:],
                      log_probs[:, start:], advantages[:, start:], action_mask[:, start:])
loss2 = actor_loss_fn(actor_log_prob[:, start:],
                      log_probs[:, start:], advantages[:, start:], action_mask[:, start-1:-1])

print('loss with mask[:, start:]:', loss1.item())
print('loss with mask[:, start-1:-1]:', loss2.item())