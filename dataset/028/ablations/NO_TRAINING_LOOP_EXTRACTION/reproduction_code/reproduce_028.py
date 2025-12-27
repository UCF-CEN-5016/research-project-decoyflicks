import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Modify num_classes to 16 instead of 10
num_classes = 16

# Assuming the rest of the model training code is in main.py
# Run the training script
# python main.py