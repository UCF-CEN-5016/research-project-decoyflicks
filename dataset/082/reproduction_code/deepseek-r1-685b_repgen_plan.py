import torch
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

# Create a minimal ViT model configuration
def create_vit_model():
    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=2,  # Reduced depth for minimal reproduction
        heads=2,  # Fewer heads
        mlp_dim=2048,
    )
    return model

# Create an MPP trainer with the same config as the model
def create_mpp_trainer(model):
    mpp_trainer = MPP(
        transformer=model,
        patch_size=32,
        dim=1024,
        mask_prob=0.15,
        random_patch_prob=0.30,
        replace_prob=0.50,
    )
    return mpp_trainer

# Sample smaller batch to make error clearer
def sample_images(batch_size):
    return torch.randn(batch_size, 3, 256, 256)

# Define a function to handle the error
def handle_error(error):
    print("Error occurred:")
    print(error)
    print("\nThe issue occurs because:")
    print("1. The patch embeddings produce [batch, num_patches, patch_dim]")
    print("2. But layer_norm expects last dimension to match model dim (1024)")
    print("3. Actual patch_dim is 3*32*32=3072 which doesn't match model dim")

# Main function
def main():
    model = create_vit_model()
    mpp_trainer = create_mpp_trainer(model)
    
    images = sample_images(2)  # Smaller batch size
    try:
        loss = mpp_trainer(images)
    except RuntimeError as e:
        handle_error(e)

if __name__ == "__main__":
    main()