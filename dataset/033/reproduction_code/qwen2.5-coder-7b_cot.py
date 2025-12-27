import torch
from torch import nn

class SE3TransformerPooled(nn.Module):
    def __init__(self, fiber_in, fiber_out, hidden_size=64):
        super().__init__()
        # Keep an attention layer reference to mirror original structure (not strictly used)
        self.slf_attn = nn.MultiheadAttention(hidden_size, 8)
        # Hidden size used for projecting different input types into a common space
        self.hidden_size = hidden_size

    def _project_to_hidden(self, t):
        # Collapse all non-batch dimensions to a single scalar per sample, then expand to hidden_size
        if t is None:
            raise ValueError("Input tensor is None")
        if t.ndim == 0:
            t = t.unsqueeze(0)
        # Ensure batch is the first dimension
        B = t.shape[0]
        flat = t.reshape(B, -1)  # (B, *)
        scalar = flat.mean(dim=1, keepdim=True)  # (B, 1)
        out = scalar.unsqueeze(1).expand(B, 1, self.hidden_size)  # (B, 1, hidden_size)
        return out

    def forward(self, x):
        # Expect x to be a dict of node type -> tensor (batch(first dim), ...)
        # Project each type into (B, 1, hidden_size)
        projected = []
        # Keep deterministic ordering of types
        for k in sorted(x.keys()):
            projected.append(self._project_to_hidden(x[k]))

        # Concatenate along the "type" dimension (dim=1). If multiple types exist,
        # the second dimension will become >1.
        concatenated = torch.cat(projected, dim=1)  # (B, n_types, hidden)

        # Use the first type as "query" (shape (B,1,hidden))
        query = projected[0]  # (B, 1, hidden)

        # Attempt a batch matmul between query and concatenated keys.
        # This will raise a batch-dimension mismatch error when n_types != 1,
        # reproducing the reported multiplication error.
        result = torch.matmul(query, concatenated.transpose(-1, -2))  # (B, 1, n_types) or raises

        return resul