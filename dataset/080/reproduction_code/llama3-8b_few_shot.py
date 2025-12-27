import torch
from vit_pytorch.vit_for_small_dataset import ViT

# Model and hyperparameters
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

# Sample data (assuming this is the cats and dogs dataset)
train_data = ...  # load training data
val_data = ...  # load validation data

# Training loop that produces inconsistent accuracy
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_data:
        inputs, labels = batch
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_data:
            inputs, labels = batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_data)}, Val Acc = {(correct / len(val_data[0])):.4f}")