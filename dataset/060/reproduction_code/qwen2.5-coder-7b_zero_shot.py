import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def build_base_dataset():
    return tf.data.Dataset.from_tensor_slices({
        'images': tf.random.uniform((1, 160, 160, 3)),
        'segmentation_masks': tf.random.uniform((1, 160, 160, 1), dtype=tf.int64)
    })


def unpackage_example(example):
    return example['images'], example['segmentation_masks']


def build_augmentation_layer():
    return tf.keras.Sequential([
        RandomFlip(tf.image.FLIP_LEFT_RIGHT),
        RandomRotation(0.5),
        RandomZoom(0.2),
    ])


def make_resized_validation_dataset(dataset, batch_size=BATCH_SIZE, autotune=AUTOTUNE):
    def _resize_and_preserve_mask(item):
        return {
            'images': tf.image.resize(item['images'], (160, 160)),
            'segmentation_masks': item['segmentation_masks']
        }

    return (
        dataset
        .map(_resize_and_preserve_mask, num_parallel_calls=autotune)
        .batch(batch_size)
        .map(unpackage_example)
        .prefetch(buffer_size=autotune)
    )


def make_augmented_training_dataset(dataset, augmentation_layer, batch_size=BATCH_SIZE, autotune=AUTOTUNE):
    def _apply_augmentation(item):
        return {
            'images': augmentation_layer(item['images']),
            'segmentation_masks': item['segmentation_masks']
        }

    return (
        dataset
        .shuffle(batch_size * 2)
        .map(_apply_augmentation, num_parallel_calls=autotune)
        .batch(batch_size)
        .map(unpackage_example)
        .prefetch(buffer_size=autotune)
    )


base_dataset = build_base_dataset()
augmentation_layer = build_augmentation_layer()

resized_validation_ds = make_resized_validation_dataset(base_dataset)
augmented_training_ds = make_augmented_training_dataset(base_dataset, augmentation_layer)

resized_validation_ds
augmented_training_ds