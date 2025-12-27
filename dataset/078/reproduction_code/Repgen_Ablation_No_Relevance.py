import torch
from vit_pytorch import ViT

def test():
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(5, 3, 256, 256)

    preds = v(img)
    assert preds.shape == (5, 1000), 'correct logits outputted'

    target = torch.randn(5, 1000)
    loss = torch.nn.CrossEntropyLoss()(preds, target)
    loss.backward()

    input_grad = img.grad
    assert input_grad.shape == (5, 3, 256, 256), 'correct input gradient shape'

test()