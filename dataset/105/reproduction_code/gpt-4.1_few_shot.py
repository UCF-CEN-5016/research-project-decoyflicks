import torch
import torch.nn as nn

class DummyCodebook(nn.Module):
    def __init__(self, codebook_size=16 * 1024, dim=512):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.embed = nn.Embedding(codebook_size, dim)
        self.embed.weight.data.uniform_(-1, 1)

    def replace(self, ind, mask, sampled):
        # This line simulates the problematic in-place assignment causing shape mismatch
        # Randomly introduce off-by-one error in mask to simulate race condition
        if torch.rand(1).item() > 0.5:
            mask = mask[:-1]  # drop one element, causing shape mismatch
        # In-place assignment that can fail if shapes don't match
        self.embed.weight.data[ind][mask] = sampled

class DummyResidualVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebook = DummyCodebook()

    def forward(self, x):
        # Dummy indices and masks for replacement
        batch_size = x.size(0)
        ind = torch.randint(0, self.codebook.codebook_size, (batch_size,))
        mask = torch.ones(batch_size, dtype=torch.bool)
        sampled = torch.randn(batch_size, self.codebook.dim)
        self.codebook.replace(ind, mask, sampled)
        return x

# Simulate input
x = torch.randn(10000, 512)

vq = DummyResidualVQ()

# Run multiple times to trigger random shape mismatch
for i in range(10):
    try:
        vq(x)
        print(f"Iteration {i}: Success")
    except RuntimeError as e:
        print(f"Iteration {i}: RuntimeError caught - {e}")