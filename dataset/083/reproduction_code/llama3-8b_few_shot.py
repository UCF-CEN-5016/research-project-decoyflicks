import torch
from vit_pytorch import ViT, DeiT

# Define two models with different architectures
model1 = ViT(input_size=224, embed_dim=768)
model2 = DeiT(input_size=224, embed_dim=768)

# Load pre-trained weights for both models
model1.load_state_dict(torch.load('vit_model.pth', map_location=torch.device('cpu')))
model2.load_state_dict(torch.load('deit_model.pth', map_location=torch.device('cpu')))

# Create input data
input_data = torch.randn(1, 3, 224, 224)

# Run forward pass for both models
output1 = model1(input_data)
output2 = model2(input_data)

# Print output to show inconsistency
print("Model 1 Output:", output1)
print("Model 2 Output:", output2)