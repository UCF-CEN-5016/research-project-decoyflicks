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

    img = torch.randn(5, 3, 256, 256)

    preds = v(img)
    loss = preds.sum()
    loss.backward()

    assert isinstance(preds.grad_fn, torch._C._autograd.UnbindBackwardAutogradNestedTensor0), "Function UnbindBackwardAutogradNestedTensor0"
    assert preds.grad.shape == (5, 16, 1024), "Incorrect gradient shape"

test()