import torch
from vit_pytorch import CrossViT

model = CrossViT(
    image_size = 224,
    num_classes = 1000,
    dim = 512,
    depth = 2,
    heads = 8,
    mlp_dim = 1024
)

input_ids = torch.tensor([[1, 2, 3]])
attention_mask = torch.tensor([[0, 0, 0]])

output = model(input_ids, attention_mask=attention_mask)