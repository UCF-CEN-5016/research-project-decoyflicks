import torch
import torch.nn as nn

class RotaryValuePE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.freq = self._calculate_freq()

    def _calculate_freq(self):
        freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2, dtype=torch.float) / self.d_model))
        return freq

    def _rotate_half(self, x):
        return torch.cat((-x[:, :, 1::2], x[:, :, ::2]), dim=-1)

    def apply_rotary_emb(self, x, start_pos=0):
        seq_len = x.shape[-2]
        position = torch.arange(start_pos, start_pos + seq_len, device=x.device, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', position, self.freq)
        cos = freqs.cos()
        sin = freqs.sin()

        x_rot = x.unsqueeze(-2) * cos.unsqueeze(-1) + self._rotate_half(x).unsqueeze(-2) * sin.unsqueeze(-1)
        x_rot = x_rot * cos.unsqueeze(-1) + self._rotate_half(x_rot).unsqueeze(-2) * sin.unsqueeze(-1)
        return x_rot.squeeze(-2)

# Test case
pe = RotaryValuePE(d_model=64)
x = torch.randn(1, 10, 64)  # (batch, seq_len, d_model)
rotated = pe.apply_rotary_emb(x)

print("Original:", x[0, 0, :4])
print("Rotated:", rotated[0, 0, :4])
print("Norm difference:", torch.norm(x - rotated))