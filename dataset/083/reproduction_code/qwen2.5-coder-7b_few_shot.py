import torch

def _transpose_last_two_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Transpose the last two non-batch dimensions (dims 1 and 2)."""
    return tensor.transpose(1, 2)

def compute_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention scores.
    The key tensor is transposed to align its dimensions before matmul,
    and the result is scaled by sqrt(3.0) to match the original behavior.
    """
    q = query
    k = _transpose_last_two_dims(key)
    scale = torch.sqrt(torch.tensor(3.))
    attention = torch.matmul(q, k) / scale
    return attention

if __name__ == "__main__":
    # Simulated query, key, and value tensors with shape (batch, seq_len, embed_dim)
    query = torch.randn(1, 2, 3)  # (batch=1, seq_len=2, embed_dim=3)
    key = torch.randn(1, 3, 2)    # Corrected key shape by transposing inside the function
    value = torch.randn(1, 2, 3)  # (batch=1, seq_len=2, embed_dim=3)

    # Cross-attention computation with corrected shapes
    attention = compute_attention(query, key, value)
    print("Attention shape:", attention.shape)