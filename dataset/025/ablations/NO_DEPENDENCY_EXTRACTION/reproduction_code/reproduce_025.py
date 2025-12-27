import tensorflow as tf
import tensorflow_models as tfm
import platform

print(tf.__version__)
print(dir(tfm.vision))

try:
    tfm.vision.augment.RandAugment()
except AttributeError as e:
    print(e)

print(platform.system(), platform.release())