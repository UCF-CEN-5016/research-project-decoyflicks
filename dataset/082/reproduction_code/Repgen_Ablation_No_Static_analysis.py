import torch
from vit_pytorch import ViT, MPP

def test():
    batch_size = 20
    image_height = 256
    image_width = 256
    patch_size = 32
    num_classes = 1000
    dim = 1024
    depth = 6
    heads = 8
    mlp_dim = 2048
    dropout = 0.1
    emb_dropout = 0.1

    model = ViT(
        image_size=image_height,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout
    )

    mask_prob = 0.15
    random_patch_prob = 0.30
    replace_prob = 0.50

    mpp_trainer = MPP(transformer=model, patch_size=patch_size, dim=dim, mask_prob=mask_prob, random_patch_prob=random_patch_prob, replace_prob=replace_prob)

    opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

    def sample_unlabelled_images(batch_size, height, width):
        return torch.rand((batch_size, 3, height, width))

    images = sample_unlabelled_images(batch_size, image_height, image_width)

    for _ in range(100):
        loss = mpp_trainer(images)
        opt.zero_grad()
        loss.backward()
        opt.step()

    preds = model(torch.randn(1, 3, 256, 256))
    assert preds.shape == (1, num_classes), 'correct logits outputted'