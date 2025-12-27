import torch

class ResidualVQ(torch.nn.Module):
    def __init__(self, codebook_size, dim):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.randn(codebook_size, dim))
    
    def replace(self, batch_samples, batch_mask):
        # Simulate replace method with shape mismatch
        ind = torch.randint(0, self.embed.size(0), (9330,))  # Indices of size 9330
        mask = torch.rand(9331, dtype=torch.bool)  # Mask of size 9331
        sampled = torch.randn(9330, 512)  # Sampled data of size 9330
        # This line causes shape mismatch error
        self.embed.data[ind][mask] = sampled

# Test the code
model = ResidualVQ(codebook_size=10000, dim=512)
x = torch.randn(1, 512)
model.replace(x, x)  # Using x as batch_mask for demonstration