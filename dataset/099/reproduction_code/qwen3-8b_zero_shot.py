import torch
import transformer_engine as te
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

model = TransformerEncoder(
    TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True,
                           init_method=te.init_method_normal(std=0.02),
                           output_layer_init_method=te.init_method_normal(std=0.02),
                           fp8=True)
)

input_ids = torch.randint(0, 10000, (2, 128, 64))
targets = torch.randint(0, 10000, (2, 128, 64))

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

for _ in range(100):
    optimizer.zero_grad()
    outputs = model(input_ids)
    loss = criterion(outputs.view(-1, 10000), targets.view(-1))
    loss.backward()
    optimizer.step()
    print(loss.item())