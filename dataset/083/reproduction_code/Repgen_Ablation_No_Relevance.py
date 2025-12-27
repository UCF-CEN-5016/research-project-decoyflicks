import torch
from vit_pytorch.cross_vit import CrossViT

# Set seed for reproducibility
torch.manual_seed(42)

# Define batch size and image dimensions
batch_size = 16
image_size = 224

# Create random input data
input_data = torch.randn(batch_size, 3, image_size, image_size)

# Attempt to initialize CrossViT with various parameter combinations
# to identify the exact cause of the initialization bug
print("Testing CrossViT initialization with different parameter combinations:")

# Test 1: Basic initialization with minimal parameters
try:
    print("\nTest 1: Basic initialization")
    model1 = CrossViT(
        image_size=image_size,
        num_classes=1000
    )
    print("✓ Basic initialization successful")
except Exception as e:
    print(f"✗ Basic initialization failed: {e}")

# Test 2: Initialization with all parameters
try:
    print("\nTest 2: Full parameter initialization")
    model2 = CrossViT(
        image_size=image_size,
        num_classes=1000,
        depth=4,
        sm_dim=192,
        sm_patch_size=16,
        sm_enc_depth=2,
        sm_enc_heads=8,
        sm_enc_mlp_dim=2048,
        lg_dim=384,
        lg_patch_size=64,
        lg_enc_depth=3,
        lg_enc_heads=8,
        lg_enc_mlp_dim=2048,
        cross_attn_depth=2,
        cross_attn_heads=8,
        dropout=0.1,
        emb_dropout=0.1
    )
    print("✓ Full parameter initialization successful")
except Exception as e:
    print(f"✗ Full parameter initialization failed: {e}")

# Test 3: Initialization with incompatible parameters
try:
    print("\nTest 3: Problematic initialization")
    model3 = CrossViT(
        image_size=image_size,
        num_classes=1000,
        sm_patch_size=32,  # Intentionally using a value that might cause issues
        lg_patch_size=32   # Same patch size may cause initialization problems
    )
    print("✓ Problematic initialization successful")
except Exception as e:
    print(f"✗ Problematic initialization failed: {e}")

# If any initialization succeeded, try a forward pass
if 'model1' in locals():
    try:
        with torch.no_grad():
            predictions = model1(input_data)
        print(f"\nForward pass successful. Output shape: {predictions.shape}")
    except Exception as e:
        print(f"\nForward pass failed: {e}")