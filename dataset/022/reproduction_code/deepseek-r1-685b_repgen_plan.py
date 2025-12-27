import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

tf.random.set_random_seed(42)

def rand_augment(images, num_layers=2, magnitude=5, level_std=0.5):
    batch_size = tf.shape(images)[0]
    magnitudes_list = []
    for _ in range(num_layers):
        base_magnitude = tf.cast(magnitude, tf.float32)
        noise = tf.random.normal([batch_size], mean=0.0, stddev=1.0)
        applied_magnitude = base_magnitude + (noise * level_std)
        magnitudes_list.append(applied_magnitude)
    magnitudes = tf.stack(magnitudes_list, axis=1)
    return magnitudes

if __name__ == "__main__":
    batch_size = 4
    height, width, channels = 128, 128, 3
    dummy_images = tf.random_uniform([batch_size, height, width, channels], minval=0, maxval=255, dtype=tf.float32)

    magnitudes_output = rand_augment(dummy_images, num_layers=2, magnitude=5, level_std=0.5)

    with tf.Session() as sess:
        mags_run = sess.run(magnitudes_output)
        for i in range(mags_run.shape[1]):
            std = np.std(mags_run[:, i])
            print(f"Layer {i} magnitude std: {std}")