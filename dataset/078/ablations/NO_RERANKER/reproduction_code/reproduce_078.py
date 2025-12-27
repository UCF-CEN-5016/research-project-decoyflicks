import torch
from vit_pytorch.na_vit_nested_tensor_3d import ViT

torch.manual_seed(42)

batch_size = 5
img = torch.randn(batch_size, 3, 256, 256)

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

preds = v(img)
assert preds.shape == (batch_size, 1000), 'correct logits outputted'

target = torch.randint(0, 1000, (batch_size,))
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(preds, target)

try:
    loss.backward()
except Exception as e:
    print(e)
    for name, param in v.named_parameters():
        if param.grad is not None:
            print(f'Gradient shape for {name}: {param.grad.shape}')