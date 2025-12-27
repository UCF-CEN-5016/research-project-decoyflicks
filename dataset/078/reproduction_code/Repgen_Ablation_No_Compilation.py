import torch
from vit_pytorch.na_vit_nested_tensor_3d import ViT

def test():
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)
    assert preds.shape == (1, 1000), 'correct logits outputted'

    target = torch.randint(0, 1000, (1,))
    loss = torch.nn.functional.cross_entropy(preds, target)

    loss.backward()

    gradients = [p.grad for p in v.parameters() if p.requires_grad]
    assert gradients[0].shape == (5, 2, 1024), 'invalid gradient shape'

test()