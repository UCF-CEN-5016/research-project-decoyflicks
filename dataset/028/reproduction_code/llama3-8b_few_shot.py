import torchvision.transforms as transforms
from vision_transformer import VisionTransformer

# Define the model and data
transform = transforms.Compose([transforms.ToTensor()])
model = VisionTransformer()
cifar10_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Train the model with incorrect `num_classes` value
model.train()
for epoch in range(1):
    for batch_idx, (inputs, labels) in enumerate(cifar10_data):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

print(f"Model num_classes: {model.num_classes}")