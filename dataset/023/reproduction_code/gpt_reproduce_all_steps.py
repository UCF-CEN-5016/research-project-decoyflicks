import sys
import subprocess

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except ImportError:
    print("TensorFlow is not installed.")
    sys.exit(1)

try:
    import official
except ImportError:
    print("tf-models-official package is not installed.")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-models-official"])
    except subprocess.CalledProcessError:
        print("Failed to install tf-models-official package.")
        sys.exit(1)

try:
    from official.vision import modeling
    from official.vision.modeling.layers import nn_blocks
    from official.vision.modeling.layers import nn_layers
    print("Successfully imported from official.vision")
except ImportError as e:
    print("ImportError occurred:")
    print(e)
    sys.exit(1)

# Test usage of imported modules to ensure they are accessible
def test_nn_blocks():
    try:
        block = nn_blocks.ResidualInner(64, 1)
        print("nn_blocks.ResidualInner instance created:", block)
    except Exception as ex:
        print("Error using nn_blocks.ResidualInner:", ex)

def test_nn_layers():
    try:
        # Assuming nn_layers has MultiHeadAttention or similar
        # Use a dummy call or instantiation if possible
        print("nn_layers module is present.")
    except Exception as ex:
        print("Error using nn_layers:", ex)

test_nn_blocks()
test_nn_layers()