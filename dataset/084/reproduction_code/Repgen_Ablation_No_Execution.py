import torch
from vit import ViT

def test():
    device = "cpu"
    
    batch_size = 16
    height, width, channels = 224, 224, 3
    
    input_data = torch.randn(batch_size, channels, height, width)
    labels = torch.randint(0, 1000, (batch_size,))
    
    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        inputs = input_data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {acc.item()}')
    
    assert acc.item() == 1.0000