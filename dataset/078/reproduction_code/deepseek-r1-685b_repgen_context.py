import torch
from vit_pytorch.na_vit_nested_tensor_3d import NaViT

def setup_model():
    model = NaViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048
    ).cuda()
    return model

def create_nested_tensor_input(batch_size):
    images = [torch.randn(3, 256, 256).cuda() for _ in range(batch_size)]
    return images

def main():
    model = setup_model()
    batch_size = 5
    images = create_nested_tensor_input(batch_size)
    outputs = model(images)

    target = torch.randint(0, 1000, (batch_size,)).cuda()
    
    loss = torch.nn.functional.cross_entropy(outputs, target)
    loss.backward()

    print("Backward pass completed")

if __name__ == "__main__":
    main()