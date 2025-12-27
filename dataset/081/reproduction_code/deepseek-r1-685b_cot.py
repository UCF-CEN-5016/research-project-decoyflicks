import torch
from vit_pytorch.cross_vit import CrossViT

# Reproduction of the bug
def reproduce_bug():
    # Original model expects 3 channels
    v = CrossViT(
        image_size=256,
        num_classes=1000,
        depth=4,
        sm_dim=192,
        sm_patch_size=16,
        lg_dim=384,
        lg_patch_size=64
    )
    
    # Try with 1 channel image
    img = torch.randn(1, 1, 256, 256)
    try:
        pred = v(img)  # This will raise the error
    except RuntimeError as e:
        print("Error reproduced successfully!")
        print(f"Error message: {str(e)}")

# Fixed version
class FixedCrossViT(CrossViT):
    def __init__(self, *args, channels=3, **kwargs):
        super().__init__(*args, **kwargs)
        # Store channels for reference
        self.channels = channels
        
        # Reinitialize embedders with correct channel count
        self.sm_image_embedder = ImageEmbedder(
            dim=self.sm_dim,
            image_size=self.image_size,
            patch_size=self.sm_patch_size,
            channels=channels,
            dropout=self.emb_dropout
        )
        self.lg_image_embedder = ImageEmbedder(
            dim=self.lg_dim,
            image_size=self.image_size,
            patch_size=self.lg_patch_size,
            channels=channels,
            dropout=self.emb_dropout
        )

def demonstrate_fix():
    # Now works with 1 channel
    v = FixedCrossViT(
        image_size=256,
        num_classes=1000,
        depth=4,
        sm_dim=192,
        sm_patch_size=16,
        lg_dim=384,
        lg_patch_size=64,
        channels=1  # Specify input channels
    )
    
    img = torch.randn(1, 1, 256, 256)
    pred = v(img)  # This will work
    print("Fixed version works! Output shape:", pred.shape)

if __name__ == "__main__":
    reproduce_bug()
    demonstrate_fix()