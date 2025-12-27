import torch

class ActorLossCalculator:
    def __init__(self):
        pass

    def actor_loss_fn(self, actor_log_prob, log_probs, advantages, action_mask):
        # This is where the bug is introduced
        actor_loss = ...  # irrelevant for this example
        return actor_loss

# Test code
calculator = ActorLossCalculator()
actor_log_prob = torch.randn(10, 100)  # irrelevant for this example
log_probs = torch.randn(10, 100)  # irrelevant for this example
advantages = torch.randn(10, 100)  # irrelevant for this example
action_mask = torch.zeros(10, 100).bool()  # mask with eos token at the end

# Calculate actor loss with incorrect masking
actor_loss = calculator.actor_loss_fn(actor_log_prob[:, 0:], log_probs[:, 0:], advantages[:, 0:], action_mask[:, 1:])  # note the slice here!

print("Actor Loss:", actor_loss)