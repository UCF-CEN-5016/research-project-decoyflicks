import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming ImageEmbedder and MultiScaleEncoder are defined in the vit_pytorch module
class CrossViT(nn.Module):
    def __init__(self, *, image_size, num_classes, sm_dim, lg_dim, sm_patch_size=12, lg_patch_size=32, sm_enc_depth=1, lg_enc_depth=4, cross_attn_depth=2):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim=sm_dim, image_size=image_size, patch_size=sm_patch_size)
        self.lg_image_embedder = ImageEmbedder(dim=lg_dim, image_size=image_size, patch_size=lg_patch_size)
        self.multi_scale_encoder = MultiScaleEncoder(sm_dim=sm_dim, lg_dim=lg_dim, cross_attn_depth=cross_attn_depth)
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)
        sm_cls = sm_tokens[:, 0]
        sm_logits = self.sm_mlp_head(sm_cls)
        return sm_logits  # Bug introduced: returning only small model logits

# Reproduction steps
batch_size = 4
image_size = (128, 128)
input_data = torch.randn(batch_size, 3, *image_size)

model = CrossViT(image_size=image_size, num_classes=10, sm_dim=64, lg_dim=128, sm_patch_size=16, lg_patch_size=32, sm_enc_depth=1, lg_enc_depth=2, cross_attn_depth=2)
output_logits = model(input_data)

assert output_logits.shape == (batch_size, 10)

# Original model output for comparison
original_model = CrossViT(image_size=image_size, num_classes=10, sm_dim=64, lg_dim=128, sm_patch_size=16, lg_patch_size=32, sm_enc_depth=1, lg_enc_depth=2, cross_attn_depth=2)
original_output_logits = original_model(input_data)

assert output_logits.shape == (batch_size, 10)
assert not torch.equal(output_logits, original_output_logits)

print("Modified output logits:", output_logits)
print("Original output logits:", original_output_logits)