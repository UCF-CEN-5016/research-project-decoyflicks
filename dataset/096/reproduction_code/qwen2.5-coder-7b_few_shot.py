import torch
from typing import Tuple

class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim: int = 10, output_dim: int = 10) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

def create_sample_batch(batch_size: int = 32, features: int = 10, num_classes: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randn(batch_size, features)
    targets = torch.randint(0, num_classes, (batch_size,))
    return inputs, targets

model = SimpleModel()
input_tensor, targets = create_sample_batch()

with torch.no_grad():
    predictions = model.predict(input_tensor.clone().detach())
    # You can now perform operations that do not require gradients
    # For example, you could do:
    # loss = some_loss_function(predictions, targets)