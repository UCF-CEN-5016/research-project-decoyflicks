import torch
from vit_pytorch import ViT
from vit_pytorch.efficient import Linformer

# Setup minimal environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create suspiciously simple dataset
class TrivialDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        # Generate two easily separable classes
        if idx % 2 == 0:
            return torch.rand(3, 224, 224), torch.tensor(0)
        return torch.rand(3, 224, 224) + 0.5, torch.tensor(1)

# Model setup matching bug report
efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# Training setup
dataset = TrivialDataset()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop that will show the bug
for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    print(f'Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} - Acc: {correct/total:.4f}')