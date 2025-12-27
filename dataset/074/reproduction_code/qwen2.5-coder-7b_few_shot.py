import torch

# Example sequence of length 5
batch_size = 1
sequence_length = 5
# Deterministic values for clarity
actor_log_prob = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
log_probs = torch.randn(batch_size, sequence_length)
# Put the "ground reward" related advantage at index 3 (the token before EOS in many setups)
advantages = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]])
action_mask = torch.tensor([
    [True, True, True, True, False],  # Index 4 is EOS (False)
], dtype=torch.bool)

start = 3  # Assume the actions start at index 3

# Masks shown in the bug discussion
original_mask = action_mask[:, start:]
print("Original mask (action_mask[:, start:]):", original_mask)

suggested_mask = action_mask[:, start-1:-1]
print("Suggested mask (action_mask[:, start-1:-1]):", suggested_mask)

correct_mask = action_mask[:, start-1:]
print("Correct mask (action_mask[:, start-1:]):", correct_mask)

# Slices as typically used in actor loss calculation (as in the bug report)
actor_slice = actor_log_prob[:, start:]          # indices [3, 4]
adv_slice = advantages[:, start-1:-1]           # indices [2, 3] -> aligns advantage[3] with actor_log_prob[4]
print("actor_slice (indices 3,4):", actor_slice)
print("adv_slice  (indices 2,3):", adv_slice)

# Compute two losses: one using the original mask (as reported) and one using the suggested mask.
loss_with_original_mask = - (actor_slice * adv_slice * original_mask.float()).sum()
loss_with_suggested_mask = - (actor_slice * adv_slice * suggested_mask.float()).sum()

print("Loss using original mask (actor_loss_fn(..., action_mask[:, start:])):", loss_with_original_mask.item())
print("Loss using suggested mask (actor_loss_fn(..., action_mask[:, start-1:-1])):", loss_with_suggested_mask.item())

# Show the elementwise products to see which term is being filtered out
elem_product = actor_slice * adv_slice
print("Elementwise actor*adv (before masking):", elem_product)
print("Elementwise after original mask:", (elem_product * original_mask.float()))
print("Elementwise after suggested mask:", (elem_product * suggested_mask.float()))