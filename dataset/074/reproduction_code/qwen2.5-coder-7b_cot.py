import torch
from transformers import AutoModelForCausalLM

def create_dummy_model():
    # Keep the original instantiation (may require actual model weights in real use)
    return AutoModelForCausalLM()

def create_dummy_batch(batch_size: int, sequence_length: int, device: torch.device = None):
    actor_log_probs = torch.randn(batch_size, sequence_length, device=device)
    log_probs = torch.randn(batch_size, sequence_length, device=device)
    advantages = torch.randn(batch_size, sequence_length, device=device)
    action_mask = torch.ones(batch_size, sequence_length, device=device)
    return actor_log_probs, log_probs, advantages, action_mask

def compute_mask_for_start(action_mask: torch.Tensor, start_index: int) -> torch.Tensor:
    # Apply the suggested change to include 'eos'
    if start_index > 0:
        return action_mask[:, start_index - 1:-1]
    return action_mask[:, :]

def main():
    model = create_dummy_model()

    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 50

    actor_log_probs, log_probs, advantages, action_mask = create_dummy_batch(
        BATCH_SIZE, SEQUENCE_LENGTH
    )

    start = 5  # Example start index
    mask = compute_mask_for_start(action_mask, start)

    actor_loss = model.actor_loss_fn(
        actor_log_probs[:, start:],
        log_probs[:, start:],
        advantages,
        mask
    )

    print("Adjusted Mask Shape:", mask.shape)

if __name__ == "__main__":
    main()