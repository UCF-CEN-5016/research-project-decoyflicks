import torch
import math

def value_pe(q_value: torch.Tensor,
             k_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute value embeddings with a single rotation for both query and key values.
    
    Args:
        q_value: Query value tensor to compute rotated embeddings for (float tensor)
        k_value: Key value tensor to compute rotated embeddings for
    
    Returns:
        A tuple containing the computed embeddings
    """
    if len(q_value) == 0:
        return q_value, k_value

    theta = math.pi / q_value.size(1)

    # Single rotation matrix applied once (correct approach)
    R_theta = torch.tensor([[math.cos(theta), -math.sin(theta)],
                            [math.sin(theta), math.cos(theta)]], dtype=q_value.dtype)
    
    # Apply rotation to both query and key values
    rotated_q = torch.mm(q_value, R_theta)
    rotated_k = torch.mm(k_value, R_theta)

    return rotated_q, rotated_k

# Example usage:
if __name__ == "__main__":
    import doctest

    random_seed = 42
    torch.manual_seed(random_seed)

    batch_size = 1
    dim = 512
    sequence_length = 64

    # Create random input tensors
    q_value = torch.randn(batch_size, sequence_length, dim)
    k_value = torch.randn(batch_size, sequence_length, dim)

    try:
        # Compute value embeddings with corrected rotation
        rotated_q, rotated_k = value_pe(q_value, k_value)

        print(f"Computed Rotations:\nShape of rotated_q: {rotated_q.shape}")
        print(f"Shape of rotated_k: {rotated_k.shape}")
        print("Example output to verify correctness.")
    except Exception as e:
        print(f"An error occurred during computation: {e}")