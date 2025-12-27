import torch

class ResidualVQ(torch.nn.Module):
    def __init__(self, codebook_size, dim):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.randn(codebook_size, dim))
    
    def replace(self, sampled, mask):
        # Ensure the shapes match
        if sampled.size(0) != mask.size(0):
            raise ValueError("Sampled data size does not match mask size")
        
        ind = torch.randint(0, self.embed.size(0), (sampled.size(0),))
        mask = mask.type(torch.bool)
        self.embed.data[ind][mask] = sampled

# Test the code
model = ResidualVQ(codebook_size=10000, dim=512)
x = torch.randn(9330, 512)
mask = torch.rand(9330, dtype=torch.bool)
model.replace(x, mask)