import tensorflow_models as tf_models


def create_rand_augment():
    """Instantiate and return a RandAugment object from tensorflow_models."""
    return tf_models.vision.augment.RandAugment()


rand_augment = create_rand_augment()