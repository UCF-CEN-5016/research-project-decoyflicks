import tensorflow as tf
from tensorflow_models.vision import augment

batch_size = 10
height, width = 299, 299
input_data = tf.random.uniform((batch_size, height, width, 3))

try:
    augment.RandAugment()
except AttributeError as e:
    assert "module 'tensorflow_models.vision' has no attribute 'augment'" in str(e)