import torch
from torch import nn
from torch.optim import Adam
from transformer_engine.pytorch import MixedPrecisionTrainer
from transformer_engine.recipe import fp8_train_recipe

# Define a minimal transformer-like model
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.embed = nn.Embedding(1000, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4),
            num_layers=2
        )
        self.output_proj = nn.Linear(64, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.output_proj(x.mean(dim=1))

# Initialize model and optimizer
model = SimpleTransformer()
optimizer = Adam(model.parameters(), lr=1e-3)

# Dummy data for training
input_ids = torch.randint(0, 1000, (32, 16))  # Batch of 32 sequences, 16 tokens
targets = torch.randint(0, 10, (32,))

# Set up FP8 training with Transformer Engine
mp_trainer = MixedPrecisionTrainer(
    model=model,
    optimizer=optimizer,
    gradient_accumulation_steps=1,
    fp8=True,  # Enable FP8 training
    fp8_recipe=fp8_train_recipe
)

# Training loop that produces NaN loss
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(input_ids)
    loss = nn.functional.cross_entropy(outputs, targets)
    mp_trainer.backward(loss)
    mp_trainer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")