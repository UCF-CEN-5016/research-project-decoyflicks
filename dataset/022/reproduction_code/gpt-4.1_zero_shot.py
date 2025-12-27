import tensorflow as tf
import numpy as np

def distort_randaugment_level(level, magnitude_std=0.5):
    level_std = tf.convert_to_tensor(magnitude_std, dtype=tf.float32)
    level = tf.cast(level, tf.float32)
    level += tf.random.normal([], 0, 1)
    return level

print(distort_randaugment_level(10, 0.5).numpy())
print(distort_randaugment_level(10, 0.0).numpy())
print(distort_randaugment_level(10, 2.0).numpy())