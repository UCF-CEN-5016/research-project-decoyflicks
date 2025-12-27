import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor

model = ViTForImageClassification("vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor("vit-base-patch16-224")

input_ids = torch.tensor([[1, 2, 3]])
attention_mask = torch.tensor([[0, 0, 0]])

output = model(input_ids, attention_mask=attention_mask)