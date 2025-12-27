import torch
from vit_pytorch import ViT  # Assuming this is the correct import based on the context

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

try:
    preds.backward(torch.ones_like(preds))
except Exception as e:
    if "Function UnbindBackwardAutogradNestedTensor0 returned an invalid gradient" in str(e):
        print("Caught expected backward pass error:", e)
    else:
        raise e