import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSimVQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, quantize_dropout=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.quantize_dropout = quantize_dropout

    def forward(self, x, return_loss=False):
        # Simulate residual connection and quantization
        x = x + self.embedding.weight  # Simplified residual connection
        
        # Quantize with dropout if enabled
        if self.quantize_dropout and self.training:
            # This line would raise NameError if return_loss is not defined
            should_quantize_dropout = self.training and self.quantize_dropout and not return_loss
            
            # Simulate dropout and shape inconsistency
            if should_quantize_dropout:
                x = F.dropout(x, p=0.5, training=self.training)
                # Intentionally create shape inconsistency for demonstration
                x = x[:, :, :1024]  # Truncate to 1024 dimensions
            else:
                x = x[:, :, 1024:]  # Take from 1024 onwards
        
        return x

# Reproduction of the bug
model = ResidualSimVQ(num_embeddings=100, embedding_dim=2048, quantize_dropout=True)
input_tensor = torch.randn(2, 17, 2048)  # (batch, timesteps, dim)

# Forward pass that triggers the bug
output = model(input_tensor)
print("Output shape:", output.shape)