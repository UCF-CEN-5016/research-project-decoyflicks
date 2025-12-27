import torch
from vit_pytorch.na_vit_nested_tensor_3d import ViT

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

    v.train()
    batch_size = 1
    img = torch.randn(batch_size, 3, 256, 256)
    preds = v(img)

    target = torch.ones((batch_size, num_classes))
    loss = torch.nn.CrossEntropyLoss()(preds, target.argmax(dim=1))

    loss.backward()

    assert v.patch_embed.proj.weight.grad.shape == (1024, 3, 32, 32), 'invalid gradient at index 0'

test()