import torch
import torch.nn as nn
from vit_pytorch.cross_vit import CrossViT  # Assuming CrossViT is defined in cross_vit.py

torch.manual_seed(42)

batch_size = 8
height, width = 224, 224
input_data = torch.randn(batch_size, 3, height, width)

model = CrossViT(
    image_size=224,
    num_classes=10,
    sm_dim=64,
    lg_dim=128,
    sm_patch_size=12,
    lg_patch_size=16,
    sm_enc_depth=1,
    lg_enc_depth=4,
    cross_attn_depth=2,
    dropout=0.1,
    emb_dropout=0.1
)

def modified_forward(self, img):
    sm_tokens = self.sm_image_embedder(img)
    lg_tokens = self.lg_image_embedder(img)
    sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)
    sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))
    sm_logits = self.sm_mlp_head(sm_cls)
    lg_logits = self.lg_mlp_head(lg_cls)
    return sm_logits + lg_logits + 1e10

CrossViT.forward = modified_forward

output = model(input_data)

assert output.shape == (batch_size, 10)
assert not torch.isnan(output).any()
assert (output > 1e10).any()