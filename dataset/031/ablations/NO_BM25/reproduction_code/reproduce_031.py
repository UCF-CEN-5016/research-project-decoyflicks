import os
import torch
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(pretrained=False).to(device)
model.eval()

weights_path = 'nvidia_resnet50_200821.pth.tar'
weights = torch.load(weights_path, map_location=device)

dummy_input = torch.empty(256, 3, 224, 224).to(device)

def load_model_weights(model, weights):
    try:
        model.load_state_dict(weights['state_dict'], strict=True)
    except RuntimeError as e:
        print(e)
        return e

error = load_model_weights(model, weights)
assert 'Missing key(s) in state_dict' in str(error) and 'Unexpected key(s) in state_dict' in str(error)

# Run the inference command
os.system("python ./main.py --arch=resnet50 --evaluate --epochs=1 --pretrained-from-file=nvidia_resnet50_200821.pth.tar -b=256 /data/imagenet/")