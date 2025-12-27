import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize_with_crop_or_pad(image, 160, 160)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

def unpackage_inputs(image, label):
    return {'image': image}, label

train_ds = tf.data.Dataset.from_tensor_slices(
    ((tf.random.uniform((100, 224, 224, 3)), tf.random.uniform((100,), minval=0, maxval=2, dtype=tf.int32)))
)

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

for batch in augmented_train_ds:
    pass