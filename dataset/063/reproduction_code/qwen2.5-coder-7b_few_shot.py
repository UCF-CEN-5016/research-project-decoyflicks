import tensorflow as tf
from typing import Any

BATCH_SIZE = 16
SHUFFLE_BUFFER = 2 * BATCH_SIZE

# Dummy dataset with int64 images
raw_images = tf.constant([[[128, 255], [0, 128]], [[255, 0], [128, 0]]], dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices(raw_images)


def cast_to_float32(image: tf.Tensor) -> tf.Tensor:
    """Cast image tensor to tf.float32."""
    return tf.cast(image, tf.float32)


def pass_through(image: tf.Tensor) -> tf.Tensor:
    """Return the input tensor unchanged (keeps original behavior)."""
    return image


pipeline = (
    dataset.shuffle(SHUFFLE_BUFFER)  # BATCH_SIZE = 16
    .map(cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(pass_through)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

if __name__ == "__main__":
    for batch in pipeline:
        print(batch)