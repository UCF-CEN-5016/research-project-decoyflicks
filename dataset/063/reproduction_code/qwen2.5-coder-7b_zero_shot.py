import tensorflow as tf

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def _identity(element: tf.Tensor) -> tf.Tensor:
    return element


def _random_crop_augment(image: tf.Tensor) -> tf.Tensor:
    return tf.image.random_crop(image, size=(150, 150, 1), seed=42)


def build_augmented_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = BATCH_SIZE,
    autotune: tf.data.AUTOTUNE = AUTOTUNE,
) -> tf.data.Dataset:
    return (
        dataset.shuffle(batch_size * 2)
        .map(_random_crop_augment, num_parallel_calls=autotune)
        .batch(batch_size)
        .map(_identity)
        .prefetch(buffer_size=autotune)
    )


source_dataset = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform((10, 160, 160, 1), minval=0, maxval=255, dtype=tf.float32)
)

augmented_dataset = build_augmented_dataset(source_dataset)