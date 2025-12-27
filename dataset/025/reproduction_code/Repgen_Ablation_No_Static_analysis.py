import tensorflow as tf
from tensorflow_models.vision import augment

batch_size = 32
input_data = tf.random.uniform((batch_size, 224, 224, 3), minval=0, maxval=1, dtype=tf.float32)

rand_augment = augment.RandAugment()
output_data = rand_augment(input_data)