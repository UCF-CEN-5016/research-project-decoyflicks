import os
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

NUM_CLASSES = 4
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

(train_ds, valid_ds, test_ds) = tfds.load(
    "oxford_iiit_pet",
    split=["train[:85%]", "train[85%:]", "test"],
    batch_size=BATCH_SIZE,
    shuffle_files=True,
)

def unpack_resize_data(section):
    image = section["image"]
    segmentation_mask = section["segmentation_mask"]
    resize_layer = keras.layers.Resizing(INPUT_HEIGHT, INPUT_WIDTH)
    image = resize_layer(image)
    segmentation_mask = resize_layer(segmentation_mask)
    return image, segmentation_mask

train_ds = train_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)

def augment_fn(image, mask):
    # Example augmentation that causes type mismatch
    image = tf.image.random_flip_left_right(image)
    mask = tf.image.random_flip_left_right(mask)
    return image, tf.cast(mask, tf.float32)

augmented_train_ds = (
    train_ds.shuffle(buffer_size=BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

for images, masks in augmented_train_ds.take(1):
    print(images.shape, masks.shape)