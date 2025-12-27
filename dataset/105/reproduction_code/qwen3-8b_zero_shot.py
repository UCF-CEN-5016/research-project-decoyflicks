import torch

class ResidualVQ:
    def __init__(self):
        self.embed = torch.nn.Parameter(torch.rand(100, 512))

    def replace(self, batch_samples, batch_mask):
        indices = torch.arange(9331)
        mask = batch_mask
        self.embed.data[indices][mask] = batch_samples

vq = ResidualVQ()
batch_samples = torch.randn(9330, 512)
batch_mask = torch.rand(9330, dtype=torch.bool)
vq.replace(batch_samples, batch_mask)