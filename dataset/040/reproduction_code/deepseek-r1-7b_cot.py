def attention(self, x):
    # ... previous layers ...
    q = self.to_q(x)
    k = self.to_k(x)
    v = x

    # Compute einsum for attention scores: [B*H*W, B*H*W]
    attn = torch.einsum('ijkl,mnop->impjnq', q, k_transposed) / (k.size(-1)**0.5)

    # Reshape and apply softmax correctly
    B, H, W, D = x.shape  # Assuming x is [B, H*W, D]
    attn = attn.view(B * H * W, -1).softmax(dim=-1)
    
    return torch.einsum('impjnq,ijkl->ijmnkl', attn, v)