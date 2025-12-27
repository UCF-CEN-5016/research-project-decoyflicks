import torch
from resnet50v1_5 import resnet50  # Ensure this path is correct for your project structure

def load_pretrained_weights(model_path):
    model = resnet50()
    checkpoint = torch.load(model_path)
    state_dict = {key.replace('module.', '') if key.startswith('module.') else key: value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    pretrained_path = 'nvidia_resnet50_200821.pth.tar'
    model = load_pretrained_weights(pretrained_path)
    print("Model loaded successfully")