import torch

class RefactoredAttention:
    def __init__(self, to_q, to_k):
        self.to_q = to_q
        self.to_k = to_k

    def attention(self, x):
        # Compute query, key, value
        queries = self.to_q(x)
        keys = self.to_k(x)
        values = x

        # Keep a transposed-version placeholder (matches original usage)
        keys_transposed = keys

        # Compute raw attention scores using einsum and scale
        attn_scores = torch.einsum('ijkl,mnop->impjnq', queries, keys_transposed) / (keys.size(-1) ** 0.5)

        # Reshape and apply softmax
        B, H, W, D = x.shape  # Preserving original shape assumption
        attn = attn_scores.view(B * H * W, -1).softmax(dim=-1)

        # Compute final output via einsum
        return torch.einsum('impjnq,ijkl->ijmnkl', attn, values)