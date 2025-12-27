import tensorflow as tf
import numpy as np

# Set up minimal environment
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Create a sample dataset with images and masks
def create_sample_dataset():
    images = np.random.rand(100, 160, 160, 3).astype(np.float32)
    masks = np.random.randint(0, 2, size=(100, 160, 160, 1)).astype(np.int64)
    return tf.data.Dataset.from_tensor_slices((images, masks))

# Define an augmentation function that applies random transformations
def augment_fn(image, mask):
    # Apply a random transformation that changes the data type of the mask tensor
    mask = tf.cast(mask, tf.float32)
    mask = tf.image.random_brightness(mask, 0.2)
    mask = tf.cast(mask, tf.int64)
    return image, mask

# Create a tf.data.Dataset from the sample dataset and apply the augmentation function
def create_augmented_dataset():
    train_ds = create_sample_dataset()
    augmented_train_ds = (
        train_ds.shuffle(BATCH_SIZE * 2)
        .map(augment_fn, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=AUTOTUNE)
    )
    return augmented_train_ds

# Trigger the bug
augmented_train_ds = create_augmented_dataset()
for batch in augmented_train_ds.take(1):
    image, mask = batch
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Mask dtype: {mask.dtype}")