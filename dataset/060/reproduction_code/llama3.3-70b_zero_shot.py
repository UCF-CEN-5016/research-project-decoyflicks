import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(inputs):
    images = inputs['images']
    segmentation_masks = inputs['segmentation_masks']
    images = tf.cast(images, tf.float32) / 255.0
    segmentation_masks = tf.cast(segmentation_masks, tf.float32)
    rand_augment = keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.5)
    outputs = rand_augment({'images': images, 'segmentation_masks': segmentation_masks})
    return outputs

def unpackage_inputs(inputs):
    images = inputs['images']
    segmentation_masks = inputs['segmentation_masks']
    return images, segmentation_masks

def resize_fn(inputs):
    images = inputs['images']
    segmentation_masks = inputs['segmentation_masks']
    images = tf.image.resize(images, (160, 160))
    segmentation_masks = tf.image.resize(segmentation_masks, (160, 160), method='nearest')
    return {'images': images, 'segmentation_masks': segmentation_masks}

train_ds = tf.data.Dataset.from_tensor_slices(
    ({
        'images': np.random.randint(0, 256, size=(100, 256, 256, 3), dtype=np.int64),
        'segmentation_masks': np.random.randint(0, 2, size=(100, 256, 256, 1), dtype=np.int64)
    })
)

val_ds = tf.data.Dataset.from_tensor_slices(
    ({
        'images': np.random.randint(0, 256, size=(100, 256, 256, 3), dtype=np.int64),
        'segmentation_masks': np.random.randint(0, 2, size=(100, 256, 256, 1), dtype=np.int64)
    })
)

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

for images, masks in augmented_train_ds:
    print(images.dtype, masks.dtype)
    break