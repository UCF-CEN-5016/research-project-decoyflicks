import torch
from vit_pytorch.vit_for_small_dataset import ViT

def test():
    batch_size = 64
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=2,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(batch_size, 3, 224, 224)
    
    # Assuming labels are random integers between 0 and num_classes-1
    labels = torch.randint(0, model.num_classes, (batch_size,))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        # Training phase
        model.train()
        optimizer.zero_grad()
        preds = model(img)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_preds = model(img)
            val_loss = criterion(val_preds, labels)
            val_acc = (val_preds.argmax(dim=1) == labels).float().mean()
        
        print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_acc.item()}')

# Run the test function
test()