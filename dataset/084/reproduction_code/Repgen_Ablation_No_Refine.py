import torch
from efficient_transformer import ViT, Linformer

# Set seed for reproducibility
torch.manual_seed(42)

# Define batch size and image dimensions
batch_size = 16
image_size = 224
patch_size = 32

# Create random input data
img = torch.randn(batch_size, 3, image_size, image_size)

# Initialize the ViT model with parameters from the bug report
efficient_transformer = Linformer(
    dim=128,
    seq_len=image_size // patch_size ** 2 + 1,  
    depth=12,
    heads=8,
    k=64
)
model = ViT(
    dim=128,
    image_size=image_size,
    patch_size=patch_size,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Define dummy target tensor
targets = torch.randint(0, 2, (batch_size,))

# Set learning rate and create optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize loss function
criterion = torch.nn.CrossEntropyLoss()

# Train the model for 5 epochs
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(img)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    # Calculate and print accuracy
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    acc = correct / total
    print(f'Epoch : {epoch + 1} - loss : {loss.item():.4f} - acc: {acc:.4f}')