import torch
from torch import nn
from torch.optim import Adam
from transformer_engine.pytorch import MixedPrecisionTrainer
from transformer_engine.recipe import fp8_train_recipe
from typing import Tuple


class MinimalTransformer(nn.Module):
    def __init__(self) -> None:
        super(MinimalTransformer, self).__init__()
        self.embedding = nn.Embedding(1000, 64)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4),
            num_layers=2
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


def build_model_and_optimizer(lr: float = 1e-3) -> Tuple[nn.Module, torch.optim.Optimizer]:
    model = MinimalTransformer()
    optimizer = Adam(model.parameters(), lr=lr)
    return model, optimizer


def generate_dummy_batch(batch_size: int = 32, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 10, (batch_size,))
    return inputs, labels


def create_mixed_precision_trainer(model: nn.Module, optimizer: torch.optim.Optimizer) -> MixedPrecisionTrainer:
    trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        gradient_accumulation_steps=1,
        fp8=True,
        fp8_recipe=fp8_train_recipe
    )
    return trainer


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer: MixedPrecisionTrainer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 10
) -> None:
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        trainer.backward(loss)
        trainer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")


if __name__ == "__main__":
    model, optimizer = build_model_and_optimizer()
    batch_inputs, batch_labels = generate_dummy_batch()
    mp_trainer = create_mixed_precision_trainer(model, optimizer)
    train_loop(model, optimizer, mp_trainer, batch_inputs, batch_labels)