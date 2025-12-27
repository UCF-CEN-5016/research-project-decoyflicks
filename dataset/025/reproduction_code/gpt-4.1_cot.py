import tensorflow_models as tfm

print(dir(tfm.vision))  # List attributes of vision submodule

# Attempt to access augment submodule
try:
    augment = tfm.vision.augment
    print("augment module found:", augment)
    randaug = augment.RandAugment()
except AttributeError as e:
    print("AttributeError:", e)