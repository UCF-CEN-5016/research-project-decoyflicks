import torch
from vit_pytorch.na_vit_nested_tensor_3d import NaViT

def create_nested_tensor(batch_size, image_size):
    # Create nested tensor input
    images = [torch.randn(3, image_size, image_size).cuda() for _ in range(batch_size)]
    return torch.stack(images)

def train_model(model, nested_images, target):
    # Forward pass
    outputs = model(nested_images)

    # Calculate loss and perform backward pass
    loss = torch.nn.functional.cross_entropy(outputs, target)
    loss.backward()

    print("Backward pass completed")

if __name__ == "__main__":
    # Model setup
    model = NaViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048
    ).cuda()

    # Create nested tensor input
    batch_size = 5
    nested_images = create_nested_tensor(batch_size, 256)

    # Create dummy target with correct shape
    target = torch.randint(0, 1000, (batch_size,)).cuda()

    # Train the model
    train_model(model, nested_images, target)