import torch
from efficient_transformer import ViT, Linformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
image_size = 224
num_classes = 1000
dim = 128
seq_len = 49 + 1
depth = 12
heads = 8
k = 64

# Create random input data
img = torch.randn(batch_size, 3, image_size, image_size).to(device)

# Define ViT model
v = ViT(
    image_size=image_size,
    patch_size=32,
    num_classes=num_classes,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# Define Linformer transformer and attach it to the ViT model
linformer = Linformer(dim, seq_len, depth, heads, k)
v.transformer = linformer

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(v.parameters(), lr=0.001)

# Create dummy labels for the random input data (assuming all classes are correct)
labels = torch.zeros(batch_size).long().to(device)

# Training loop
for epoch in range(5):
    v.train()
    optimizer.zero_grad()
    preds = v(img)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    _, predicted = torch.max(preds.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / batch_size

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy}')
    
# Assert that the accuracy rate is 100% at least once during the training process
assert accuracy == 1.0, 'Accuracy should be 100%'