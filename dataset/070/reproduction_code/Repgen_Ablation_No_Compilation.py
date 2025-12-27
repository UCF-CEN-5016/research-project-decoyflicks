import argparse
import deepspeed
from torchvision import datasets, transforms
import torch
import torchvision.models

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_dim', type=int, default=224)
parser.add_argument('--data_path', type=str, required=True)
args = parser.parse_args()

# Initialize distributed training
deepspeed.init_distributed()

# Define transformations
transform = transforms.Compose([
    transforms.Resize(args.image_dim),
    transforms.CenterCrop(args.image_dim),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Load dataset
train_dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Define model architecture (example: ResNet)
model = torchvision.models.resnet50(pretrained=True)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up loss function
criterion = torch.nn.CrossEntropyLoss()

# Initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create deepspeed configuration
ds_config = deepspeed.add_config_arguments(parser).parse_args()

# Initialize DeepSpeed trainer
deepspeed_model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Training loop
for epoch in range(10):
    for images, target in train_loader:
        images, target = images.to(device), target.to(device)
        outputs = deepspeed_model(images)
        loss = criterion(outputs, target)
        deepspeed_model.backward(loss)
        deepspeed_model.step()