import tensorflow_models as tfm

# Trying to access augment module directly from vision raises AttributeError
tfm.vision.augment.RandAugment()