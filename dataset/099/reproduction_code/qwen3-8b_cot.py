import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_engine import transformer_engine
from transformer_engine.common import get_cuda_version, get_rocm_version
from transformer_engine.pytorch import fp8, transformer_engine

# Set up FP8 environment
fp8_recipe = fp8.FP8Recipe(
    amp_recipe=fp8.AMPRecipe(
        amp_type=fp8.AMPType.TRAINING,
        amp_dtype=fp8.AMPDType.FP8_E5M2,
        amp_layout=fp8.AMPLayout.ROW_MAJOR
    )
)

# Define a minimal transformer model
class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size=5035, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, src, tgt):
        src_emb = self.embedding(src)  # (seq_len, batch, embed_dim)
        tgt_emb = self.embedding(tgt)  # (seq_len, batch, embed_dim)
        out = self.transformer(src_emb, tgt_emb)
        out = self.fc(out)
        return out

# Set up training environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MinimalTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Synthetic data
src = torch.randint(0, 5035, (10, 32)).to(device)  # (seq_len, batch)
tgt = torch.randint(0, 5035, (10, 32)).to(device)  # (seq_len, batch)

# Training loop
for step in range(1000):
    optimizer.zero_grad()
    
    # Forward pass with FP8
    with transformer_engine.fp8_context(fp8_recipe):
        outputs = model(src, tgt)
        loss = F.cross_entropy(outputs.view(-1, 5035), tgt.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Check for NaNs
    if torch.isnan(loss):
        print(f"NaN loss detected at step {step}")
        break
    
    optimizer.step()