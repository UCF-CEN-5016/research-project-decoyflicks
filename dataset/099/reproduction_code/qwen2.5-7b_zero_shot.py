import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, fp8=True):
        super(CustomTransformerEncoderLayer, self).__init__(d_model=d_model, nhead=nhead)
        self.init_method = torch.nn.init.normal_
        self.output_layer_init_method = torch.nn.init.normal_
        self.fp8 = fp8

class CustomTransformerEncoder(TransformerEncoder):
    def __init__(self, d_model, nhead, fp8=True):
        encoder_layer = CustomTransformerEncoderLayer(d_model=d_model, nhead=nhead, fp8=fp8)
        super(CustomTransformerEncoder, self).__init__(encoder_layer, num_layers=1)

model = CustomTransformerEncoder(d_model=128, nhead=8)

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