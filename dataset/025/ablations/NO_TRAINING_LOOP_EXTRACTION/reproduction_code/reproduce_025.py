import tensorflow as tf
import tensorflow_models as tfm

print(tf.__version__)
print(tfm.vision.augment)

try:
    rand_augment = tfm.vision.augment.RandAugment()
except AttributeError as e:
    print(e)

# Check directory structure and other modules
try:
    print(tfm.vision.backbones)
    print(tfm.vision.configs)
except AttributeError as e:
    print(e)

# Log environment details
import platform
import sys

print("Operating System:", platform.mac_ver())
print("Python Version:", sys.version)

# Check TensorFlow installation
import subprocess

subprocess.run(["pip", "show", "tensorflow"])
subprocess.run(["pip", "show", "tensorflow-macos"])