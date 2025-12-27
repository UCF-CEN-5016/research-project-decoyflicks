import torch.nn as nn

x = torch.randn(2, 5)  # batch of 2 sequences, each with length 5

model = nn.Sequential(
    nn.Embedding(3, 3),
    nn.Linear(3, 4),  # expecting input dim=3 but got output_dim=4
    nn.Linear(4, 1)
)

print(f"Input tensor shape: {x.shape}")
output = model(x)
print(f"Output tensor shape after first Linear layer: {list(model[:2].parameters())[0].shape}")