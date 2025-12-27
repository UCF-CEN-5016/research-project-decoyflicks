# [Original imports remain unchanged]

def rotate_half(x):
    """Rotates x by half a dimension, i.e., an input of shape (B, H, W)
    is rotated into shape (B, W, H) where W > H."""
    # ... previous code ...

    if self.n%2 == 0:
        return x
    else:
        x_rot = rotate_half(x)
        x_1 = x_rot[:, :n//2]
        x_2 = x_rot[:, n//2:]
        return torch.cat([x_2, x_1], dim=-1)
    
value_embedding = rotate_half(value_embedding)
# Removed the second rotation line