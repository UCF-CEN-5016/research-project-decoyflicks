# Reproduction code for the augment module access bug
import tensorflow_models as tfm

# This works (other submodules are accessible)
print("Available vision submodules:", dir(tfm.vision))

# This fails
try:
    from tensorflow_models.vision import augment
    print("Successfully imported augment")
except ImportError as e:
    print(f"Import failed: {e}")

# Direct import attempt
try:
    import tensorflow_models.vision.augment as augment
    print("Direct import worked")
except ImportError as e:
    print(f"Direct import failed: {e}")

# Verify package structure
import os
from importlib.util import find_spec

spec = find_spec("tensorflow_models.vision.augment")
if spec:
    print(f"Module found at: {spec.origin}")
    print(f"File exists: {os.path.exists(spec.origin)}")
else:
    print("augment module spec not found")

# Until the package structure is fixed, use direct file import
from official.vision.ops import augment

# Now you can use RandAugment
augmenter = augment.RandAugment()
print("Created RandAugment instance:", augmenter)