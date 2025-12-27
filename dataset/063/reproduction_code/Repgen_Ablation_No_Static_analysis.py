import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
import random

BATCH_SIZE = 128
IMG_HEIGHT = 160
IMG_WIDTH = 160
NUM_CLASSES = 3

# Disable progress bar for TensorFlow Datasets
tfds.disable_progress_bar()

# Load Oxford-IIT Pet dataset split into train+test[:80%] and test[80%:]
(train_ds, val_ds), test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'path_to_oxford_iit_pet',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="both",
    seed=123
)

# Define a lambda function to rescale images and correct segmentation masks
rescale_and_correct_masks = lambda x, y: (x / 255.0, tf.where(y > 0, 1.0, 0.0))

# Map the rescaling function to the train_ds and val_ds datasets
train_ds = train_ds.map(rescale_and_correct_masks)
val_ds = val_ds.map(rescale_and_correct_masks)

# Create an augment_fn sequence with resizing, random flip, random rotation, and RandAugment
augment_fn = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    # Add more augmentation layers if needed
])

# Shuffle the train_ds dataset with buffer size of batch_size * 2
train_ds = train_ds.shuffle(BATCH_SIZE * 2)

# Apply the augment_fn function in parallel to the shuffled train_ds using num_parallel_calls=AUTOTUNE
train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Batch the augmented train_ds dataset with batch size of BATCH_SIZE
train_ds = train_ds.batch(BATCH_SIZE)

# Map the unpackage_inputs function to the batched train_ds
def unpackage_inputs(ds):
    return ds[0], ds[1]

train_ds = train_ds.map(unpackage_inputs)

# Prefetch the batched and mapped train_ds dataset with buffer size tf.data.AUTOTUNE
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# Attempt to access an element from the augmented_train_ds dataset
element = next(iter(train_ds))