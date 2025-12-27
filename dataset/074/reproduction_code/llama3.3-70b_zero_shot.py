import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorLoss(nn.Module):
    def __init__(self):
        super(ActorLoss, self).__init__()
        self.actor_loss_fn = nn.MSELoss()

    def forward(self, actor_log_prob, log_probs, advantages, action_mask, start):
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages, action_mask[:, start:])
        return actor_loss

def main():
    actor_log_prob = torch.randn(1, 10)
    log_probs = torch.randn(1, 10)
    advantages = torch.randn(1, 10)
    action_mask = torch.ones(1, 10)
    start = 1

    model = ActorLoss()
    loss = model(actor_log_prob, log_probs, advantages, action_mask, start)
    print(loss)

    actor_log_prob = torch.randn(1, 10)
    log_probs = torch.randn(1, 10)
    advantages = torch.randn(1, 10)
    action_mask = torch.ones(1, 10)
    start = 1

    actor_loss_fn = nn.MSELoss()
    actor_loss = actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages)
    print(actor_loss)

if __name__ == "__main__":
    main()