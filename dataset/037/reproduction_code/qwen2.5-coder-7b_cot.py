# [Original imports remain unchanged]

def rotate_half(tensor):
    """Rotate `tensor` by half along the last dimension.

    For an input shaped (..., 2*m) this returns concat(-second_half, first_half)
    along the last dimension. This is the standard 'rotate half' used in
    rotary positional embeddings.
    """
    a, b = tensor.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class HalfDimensionSwapper:
    """Helper to conditionally swap halves of a tensor based on an integer n."""

    def __init__(self, n):
        self.n = n

    def swap_if_odd(self, tensor):
        """If self.n is even return tensor unchanged; otherwise rotate and swap halves.

        Preserves the original logic:
            if self.n % 2 == 0:
                return tensor
            else:
                x_rot = rotate_half(tensor)
                x_1 = x_rot[:, :n//2]
                x_2 = x_rot[:, n//2:]
                return torch.cat([x_2, x_1], dim=-1)
        """
        if self.n % 2 == 0:
            return tensor

        rotated = rotate_half(tensor)
        first_half = rotated[..., : self.n // 2]
        second_half = rotated[..., self.n // 2 :]
        return torch.cat([second_half, first_half], dim=-1)


# Apply a single rotation to the value embedding (kept as in the original logic).
value_embedding = rotate_half(value_embedding)