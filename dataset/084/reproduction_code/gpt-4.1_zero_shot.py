import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class DummyTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    def forward(self, x):
        return self.linear(x)

class DummyViT(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = DummyTransformer(dim)
        self.mlp_head = nn.Linear(dim, num_classes)
    def forward(self, x):
        b, _, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        return self.mlp_head(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
dim = 128
num_classes = 2
model = DummyViT(dim=dim, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Dataset: all inputs identical, all labels identical
x = torch.randn(313*16, 49+1, dim)
x[:,1:,:] = x[:,1:,:][0]  # make all patch tokens identical
y = torch.zeros(313*16, dtype=torch.long)

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for epoch in range(1, 6):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * by.size(0)
        pred = out.argmax(1)
        correct += (pred == by).sum().item()
        total += by.size(0)
    print(f"Epoch : {epoch} - loss : {total_loss/total:.4f} - acc: {correct/total:.4f}")