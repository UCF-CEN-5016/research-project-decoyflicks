import torch
from vit_pytorch.cross_vit import CrossViT

def test_crossvit():
    v = CrossViT(
        image_size=256,
        patch_size=16,
        num_classes=10,
        dim=768
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)
    assert preds.shape == (1, 10), 'correct logits outputted'

test_crossvit()