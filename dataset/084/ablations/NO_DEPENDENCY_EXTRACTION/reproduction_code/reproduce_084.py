import torch
import torch.nn as nn
import torch.optim as optim

# Assuming ViT is defined elsewhere in the codebase
# from your_model_library import ViT  # Uncomment and replace with actual import

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

dim = 128
image_size = 224
patch_size = 32
num_classes = 2
depth = 12
heads = 8
k = 64

# Initialize the Vision Transformer model
model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, 
            dim=dim, depth=depth, heads=heads, mlp_dim=dim*4, channels=3).to(device)

batch_size = 32
# Generate random input data and target labels
input_data = torch.randn(batch_size, 3, image_size, image_size).to(device)
target_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(input_data)
    loss = loss_function(output, target_labels)
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    accuracy = (output.argmax(dim=1) == target_labels).float().mean().item()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy}')

# Validation phase
model.eval()
validation_data = torch.randn(batch_size, 3, image_size, image_size).to(device)
with torch.no_grad():
    val_output = model(validation_data)
    val_accuracy = (val_output.argmax(dim=1) == target_labels).float().mean().item()

# Assert that validation accuracy is 1.0 to reproduce the bug
assert val_accuracy == 1.0