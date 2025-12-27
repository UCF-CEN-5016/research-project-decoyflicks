import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple actor loss function
class ActorLossFn(nn.Module):
    def __init__(self):
        super(ActorLossFn, self).__init__()
    
    def forward(self, actor_log_prob, log_probs, advantages, action_mask):
        # Simplified actor loss calculation
        loss = -torch.sum(actor_log_prob * advantages * action_mask) / torch.sum(action_mask)
        return loss

# Set up minimal environment
if __name__ == "__main__":
    # Assume we have the following tensors
    actor_log_prob = torch.randn(1, 10)  # Log probabilities of actions
    log_probs = torch.randn(1, 10)  # Log probabilities of actions (for demonstration)
    advantages = torch.randn(1, 10)  # Advantages of actions
    action_mask = torch.ones(1, 10)  # Mask for actions, assuming all actions are valid initially
    action_mask[:, -1] = 0  # Set the last action as invalid (eos token)
    
    start = 1  # Starting index for slicing
    
    # Define the actor loss function
    actor_loss_fn = ActorLossFn()
    
    # Original code with the bug
    original_loss = actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages[:, start:], action_mask[:, start:])
    print(f"Original Loss: {original_loss}")
    
    # Proposed fix
    fixed_loss = actor_loss_fn(actor_log_prob[:, start-1:-1], log_probs[:, start-1:-1], advantages[:, start-1:-1], action_mask[:, start-1:-1])
    print(f"Fixed Loss: {fixed_loss}")