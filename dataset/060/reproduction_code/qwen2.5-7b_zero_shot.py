import tensorflow as tf
from tensorflow.keras import layers
import keras_cv

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 8

# Dummy data
images = tf.random.uniform(shape=(100, 160, 160, 3), dtype=tf.float32)
masks = tf.random.uniform(shape=(100, 160, 160, 1), dtype=tf.int64)

def unpackage_inputs(x):
    return x[0], x[1]

def augment_image_only(images, masks):
    return tf.image.flip_left_right(images), masks

def augment_mask_only(images, masks):
    return images, tf.cast(masks, tf.float32)

# Create dataset
train_ds = tf.data.Dataset.from_tensor_slices((images, masks)).shuffle(100)

# Apply augmentations with conflicting types
augment_fn_images = keras_cv.layers.RandAugment(
    augmentations=[
        keras_cv.layers.FlipLeftRight(),
        keras_cv.layers.RandomBrightness(0.2)
    ],
    num_magnitude_bins=10,
    apply_to='images'
)

augment_fn_masks = keras_cv.layers.RandAugment(
    augmentations=[
        keras_cv.layers.RandomContrast(0.2),
        keras_cv.layers.RandomSaturation(0.2)
    ],
    num_magnitude_bins=10,
    apply_to='masks'
)

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn_images, num_parallel_calls=AUTOTUNE)
    .map(augment_fn_masks, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)