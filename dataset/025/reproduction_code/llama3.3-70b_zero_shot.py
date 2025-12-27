import tensorflow_models as tfm

try:
    tfm.vision.augment.RandAugment()
except AttributeError as e:
    print(e)