import tensorflow_models as tfm

# Attempt to use RandAugment from vision.augment module
try:
    from tensorflow_models.vision.augment import RandAugment
    augmenter = RandAugment()
    print("Successfully created RandAugment instance")
except AttributeError as e:
    print(f"Error: {e}")
    print("Available vision submodules:", dir(tfm.vision))

# Verify other vision submodules work (for comparison)
try:
    backbones = tfm.vision.backbones
    print("Successfully accessed vision.backbones")
except AttributeError as e:
    print(f"Backbones error: {e}")