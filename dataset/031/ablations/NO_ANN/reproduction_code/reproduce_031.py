import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models  # Fixed undefined variable issue
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def main():
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.resnet50(pretrained=False)  # Use the correct import for models
    
    state_dict = torch.load('nvidia_resnet50_200821.pth.tar')['state_dict']
    
    try:
        model.load_state_dict(state_dict, strict=False)  # Allow loading with strict=False to reproduce the bug
    except RuntimeError as e:
        print(e)
        assert 'Missing key(s) in state_dict' in str(e)  # Check for missing keys
        assert 'Unexpected key(s) in state_dict' in str(e)  # Check for unexpected keys

    dataset = datasets.ImageFolder('/data/imagenet/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for inputs, labels in data_loader:
        outputs = model(inputs.to(device))  # Ensure inputs are sent to the correct device

if __name__ == '__main__':
    main()