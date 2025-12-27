# Assuming sample code where ind and mask are boolean tensors
combined_mask = (ind & mask).float()
self.embed.data[combined_mask] = sampled