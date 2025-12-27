import sys
import subprocess
import tensorflow as tf

def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}.")
        sys.exit(1)

def verify_tensorflow_installation():
    try:
        print("TensorFlow version:", tf.__version__)
    except ImportError:
        print("TensorFlow is not installed.")
        sys.exit(1)

def verify_tf_models_official_installation():
    try:
        import official
    except ImportError:
        print("tf-models-official package is not installed. Attempting to install...")
        install_package("tf-models-official")

def import_official_vision_modules():
    try:
        from official.vision import modeling
        from official.vision.modeling.layers import nn_blocks
        from official.vision.modeling.layers import nn_layers
        print("Successfully imported from official.vision")
        return modeling, nn_blocks, nn_layers
    except ImportError as e:
        print("ImportError occurred for official.vision modules:")
        print(e)
        sys.exit(1)

def test_nn_blocks(nn_blocks_module):
    try:
        block = nn_blocks_module.ResidualInner(64, 1)
        print("nn_blocks.ResidualInner instance created:", block)
    except Exception as ex:
        print("Error using nn_blocks.ResidualInner:", ex)

def test_nn_layers(nn_layers_module):
    try:
        print("nn_layers module is present.")
    except Exception as ex:
        print("Error using nn_layers:", ex)

def main():
    verify_tensorflow_installation()
    verify_tf_models_official_installation()
    
    modeling, nn_blocks, nn_layers = import_official_vision_modules()

    test_nn_blocks(nn_blocks)
    test_nn_layers(nn_layers)

if __name__ == "__main__":
    main()
