import torch
import deepspeed

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters())

# Initialize DeepSpeed engine with the model and optimizer
ds_model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=dict(train_batch_size=2, fp16=dict(enabled=True))
)

# Training loop
for _ in range(1):  # Train only one epoch
    inputs = torch.randn(2, 10)
    outputs = ds_model(inputs)
    loss = outputs.sum()
    ds_model.backward(loss)
    ds_model.step()
    
    # Correct model access without causing an error
    print_throughput(ds_model.module, ...)  # Access the wrapped model

print_throughput(ds_model.module, ...)  # Correct access to the wrapped model outside the loop