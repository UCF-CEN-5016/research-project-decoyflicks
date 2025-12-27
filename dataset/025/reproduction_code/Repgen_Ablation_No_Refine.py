import tensorflow as tf
from official.vision.augment import RandAugment

# Define a batch size of 32
batch_size = 32
height, width = 160, 160

# Create random uniform input data with shape (batch_size, height, width, 3)
input_data = tf.random.uniform((batch_size, height, width, 3), minval=0, maxval=256, dtype=tf.int32)

# Call RandAugment function with no parameters to initialize it
rand_augment = RandAugment()

# Verify AttributeError is raised when calling the initialized RandAugment object
try:
    rand_augment(input_data)
except AttributeError as e:
    print(e)