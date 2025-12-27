import tensorflow as tf
import numpy as np
from tensorflow.data import AUTOTUNE

# Minimal environment setup
train_ds = tf.data.Dataset.from_tensor_slices((np.random.rand(10, 160, 160, 1), np.random.randint(0, 256, (10,))))
BATCH_SIZE = 4

# Add triggering conditions: Use `map` twice with different data types
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(lambda x: tf.image.rand_brightness(x[1], 0.5), num_parallel_calls=AUTOTUNE)  # Int64 type
    .map(lambda x: tf.image.random_jpeg_quality(x[0], 10, 90), num_parallel_calls=AUTOTUNE)  # Float32 type
    .batch(BATCH_SIZE)
)

# Wrap final code in a try-except block to catch the error
try:
    augmented_train_ds = augmented_train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
except Exception as e:
    print(f"Error: {e}")