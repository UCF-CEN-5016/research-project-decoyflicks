import torch
from torch import nn
from x_transformers import TransformerWrapper, Decoder

class CustomTransformer(nn.Module):
    def __init__(self):
        super(CustomTransformer, self).__init__()
        self.decoder = Decoder(
            dim=512,
            depth=6,
            heads=8,
            kv_heads=4,
            qk_norm=True,
            dim_head=64
        )
        self.model = TransformerWrapper(
            num_tokens=256,
            max_seq_len=512,
            attn_layers=self.decoder
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the custom transformer model
model = CustomTransformer()

# Forward pass with sample input
x = torch.randint(0, 256, (1, 128))
out = model(x)

print(out.shape)