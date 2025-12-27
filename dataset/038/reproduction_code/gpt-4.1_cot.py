import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim  # feature dim
        self.max_seq_len = max_seq_len

        # Create cos_cached and sin_cached with shape: (max_seq_len, 1, 1, dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.einsum('i , j -> i j', position, inv_freq)
        # shape (max_seq_len, dim/2)
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()

        # Expand to shape (max_seq_len, 1, 1, dim)
        # To match input x: (seq_len, batch, heads, dim)
        # We interleave sin and cos to full dim
        sin_cos = torch.zeros(max_seq_len, dim)
        sin_cos[:, 0::2] = sin
        sin_cos[:, 1::2] = cos
        # But in the original code, cos_cached and sin_cached are stored separately
        # So we store sin and cos separately
        self.register_buffer('sin_cached', sin.unsqueeze(1).unsqueeze(1))  # (max_seq_len,1,1,dim/2)
        self.register_buffer('cos_cached', cos.unsqueeze(1).unsqueeze(1))  # (max_seq_len,1,1,dim/2)

    def forward(self, x):
        """
        x shape: (seq_len, batch, heads, dim)
        Applies rotary positional embedding to half the dim (d = dim//2)
        """
        d = self.dim // 2
        # Split x into halves along last dim
        x1 = x[..., :d]
        x2 = x[..., d:]

        # Apply rotary embedding only to x1 and x2?
        # The original formula usually applies to pairs of dimensions,
        # Here we simulate the common rotation:
        # [x1 * cos - x2 * sin, x1 * sin + x2 * cos]

        # But the reported bug is about mismatch in shape in:
        # x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        # where x_rope and neg_half_x have dim d=3, but cos_cached and sin_cached have dim 4

        # Let's simulate that condition by setting dim=4 but only applying rotary on last 3 dims
        # So for reproduction, we intentionally mismatch dim and d

        # For demonstration, assume self.sin_cached and cos_cached have dim=4 (the full dim)
        # but x1 and x2 have dim=3 (partial feature)

        # Construct sin_cached and cos_cached with dim=4 to reproduce error
        # But x has last dim=4, and we will slice x1 to dim=3 to simulate partial features

        # To reproduce the error:
        # Let's slice x1 to last dim=3, but cos_cached and sin_cached have dim=4

        # Let's proceed:

        x_rope = x[..., :3]  # partial feature dim=3
        neg_half_x = torch.cat([-x[..., 3:].unsqueeze(-1), torch.zeros_like(x[..., :2])], dim=-1)  # dummy neg_half_x to match dim=3

        # The error line (original buggy code):
        # x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        # Will fail because last dim of x_rope is 3, but cos_cached and sin_cached last dim is 4

        # Let's try to trigger that error:
        return (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

def main():
    seq_len = 5
    batch = 2
    heads = 3
    dim = 4  # feature dimension

    # Initialize module with dim=4
    rotary = RotaryPositionalEmbedding(dim=dim, max_seq_len=seq_len)

    # Create input x with shape (seq_len, batch, heads, dim)
    x = torch.randn(seq_len, batch, heads, dim)

    # Intentionally cause the partial feature dim mismatch by slicing
    # But forward expects full dim=4, so we simulate by calling forward directly
    try:
        out = rotary(x)
    except RuntimeError as e:
        print("Caught RuntimeError as expected:")
        print(e)

if __name__ == "__main__":
    main()