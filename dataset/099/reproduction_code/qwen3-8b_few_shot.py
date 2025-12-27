import torch
import transformer_engine as te
from transformer_engine.pytorch import MixedPrecisionTrainer

# Define a minimal transformer-like model
class SimpleTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(1000, 64)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=64, nhead=4),
            num_layers=2
        )
        self.output_proj = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.output_proj(x.mean(dim=1))

# Initialize model and optimizer
model = SimpleTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy data for training
input_ids = torch.randint(0, 1000, (32, 16))  # Batch of 32 sequences, 16 tokens
targets = torch.randint(0, 10, (32,))

# Set up FP8 training with Transformer Engine
mp_trainer = MixedPrecisionTrainer(
    model=model,
    optimizer=optimizer,
    gradient_accumulation_steps=1,
    fp8=True,  # Enable FP8 training
    fp8_recipe=te.recipe.fp8_train_recipe
)

# Training loop that produces NaN loss
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(input_ids)
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    mp_trainer.backward(loss)
    mp_trainer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")