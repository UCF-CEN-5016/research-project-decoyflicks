import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

tf.random.set_random_seed(42)

# Dummy RandAugment implementation snippet (simulate relevant part)
def rand_augment(images, num_layers=2, magnitude=5, level_std=0.5):
    batch_size = tf.shape(images)[0]
    magnitudes = []
    for i in range(num_layers):
        base_magnitude = tf.cast(magnitude, tf.float32)
        # Bug reproduction: magnitude noise applied as standard normal without scaling by level_std
        noise = tf.random.normal([batch_size], mean=0.0, stddev=1.0)
        applied_magnitude = base_magnitude + noise  # BUG: should multiply noise by level_std
        magnitudes.append(applied_magnitude)
    magnitudes = tf.stack(magnitudes, axis=1)
    return magnitudes

batch_size = 4
height, width, channels = 128, 128, 3
dummy_images = tf.random_uniform([batch_size, height, width, channels], minval=0, maxval=255, dtype=tf.float32)

magnitudes = rand_augment(dummy_images, num_layers=2, magnitude=5, level_std=0.5)

with tf.Session() as sess:
    mags = sess.run(magnitudes)
    for i in range(mags.shape[1]):
        std = np.std(mags[:, i])
        print(f"Layer {i} magnitude std: {std}")