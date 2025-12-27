import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Create a sample dataset with images and segmentation masks
train_dir = 'path/to/train/directory'
val_dir = 'path/to/validation/directory'

train_ds = ImageDataGenerator(
    rescale=1./255,
    batch_size=BATCH_SIZE,
).flow_from_directory(train_dir, class_mode='categorical', target_size=(160, 160))

val_ds = ImageDataGenerator(
    rescale=1./255,
).flow_from_directory(val_dir, class_mode='categorical', target_size=(160, 160))

# Define the augment function
def augment_fn(image, segmentation_mask):
    # Simulate random augmentations (e.g., rotation, flipping)
    image = tf.image.random_rotation(image, 30)
    segmentation_mask = tf.image.random_flip_left_right(segmentation_mask)
    return {'images': image, 'segmentation_masks': segmentation_mask}

# Apply the augment function to the training dataset
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)  # assumes an unpackage_inputs function exists
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Repeat the process for the validation dataset
resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Wrap everything in a cell for easier testing: