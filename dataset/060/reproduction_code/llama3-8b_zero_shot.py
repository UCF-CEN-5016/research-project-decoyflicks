import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomZoom

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices({
    'images': tf.random.uniform((1, 160, 160, 3)),
    'segmentation_masks': tf.random.uniform((1, 160, 160, 1), dtype=tf.int64)
})

augment_fn = tf.keras.Sequential([
    RandomFlip(tf.image.FLIP_LEFT_RIGHT),
    RandomRotation(0.5),
    RandomZoom(0.2),
])

resized_val_ds = (
    train_ds.map(lambda x: {'images': tf.image.resize(x['images'], (160, 160))}, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

unpackage_inputs = lambda x: (x['images'], x.pop('segmentation_masks'))

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

resized_val_ds
augmented_train_ds