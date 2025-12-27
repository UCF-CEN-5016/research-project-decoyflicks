import tensorflow as tf
import keras_cv
from keras_cv import layers as cv_layers

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(images, masks):
    # Apply random augmentations
    return images, masks

def unpackage_inputs(data):
    return data

def resize_fn(images, masks):
    return images, masks

train_images = tf.random.uniform((100, 160, 160, 3), dtype=tf.float32)
train_masks = tf.random.uniform((100, 160, 160, 1), dtype=tf.float32)
val_images = tf.random.uniform((20, 160, 160, 3), dtype=tf.float32)
val_masks = tf.random.uniform((20, 160, 160, 1), dtype=tf.float32)

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))

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

# Run the dataset creation to reproduce the bug
for _ in augmented_train_ds.take(1):
    pass

for _ in resized_val_ds.take(1):
    pass