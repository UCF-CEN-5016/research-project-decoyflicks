import tensorflow as tf
from official.vision.ops.augment import RandAugment

# Setup
batch_size = 32
height, width = 224, 224
input_data = tf.random.uniform((batch_size, height, width, 3))

# Define RandAugment function
def apply_rand_augment(level_std):
    rand_augment = RandAugment(magnitude_stddev=1.0, level_std=level_std)
    return rand_augment(input_data)

# Test with level_std = 0
output_0 = apply_rand_augment(level_std=0)
print("Output with level_std=0, stddev:", tf.math.reduce_std(output_0))

# Test with level_std = 1
output_1 = apply_rand_augment(level_std=1)
print("Output with level_std=1, stddev:", tf.math.reduce_std(output_1))

# Log outputs
print("Output values (level_std=0):", output_0.numpy())
print("Output values (level_std=1):", output_1.numpy())