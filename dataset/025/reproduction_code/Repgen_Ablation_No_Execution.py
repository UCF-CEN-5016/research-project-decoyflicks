import tensorflow_models as tfm
import numpy as np

def test_rand_augment():
    # Correctly import RandAugment from the correct submodule
    augment_instance = tfm.vision.augment.RandAugment()
    assert augment_instance is not None

test_rand_augment()