import torch
import deepspeed

# Minimal model setup
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters())
# Initialize DeepSpeed engine
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=dict(train_batch_size=2, fp16=dict(enabled=True))
)

# Training loop that triggers the error
for _ in range(1):  # Train only one epoch
    inputs = torch.randn(2, 10)
    outputs = model(inputs)
    loss = outputs.sum()
    model.backward(loss)
    model.step()
    
    # Incorrect model access that causes the error
    print_throughput(model.model, ...)  # This line would raise AttributeError

print_throughput(model.module, ...)  # Correct access to the wrapped model