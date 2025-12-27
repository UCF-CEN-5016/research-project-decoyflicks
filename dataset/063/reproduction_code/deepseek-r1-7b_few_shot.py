import tensorflow as tf

# Create a dataset with sample images and labels (simulating mixed dtypes)
images = tf.random.uniform([10, 256, 256, 3], dtype=tf.float32)  # 10 images of shape [256x256x3]
labels = tf.range(10, dtype=tf.int32)

# Combine into a dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

def augment_fn(x):
    img = x[0]  # Image tensor (float32)
    # Incorrectly cast to int64 for some reason (this is where the error might occur if not fixed)
    img = tf.cast(img, dtype=tf.int64) 
    return img

# Apply the augment function
augmented_dataset = dataset.map(augment_fn)

# Now attempt to perform operations that would trigger the dtype error

import tensorflow as tf

# Create sample images and labels (labels are in integer format)
images = tf.random.uniform([10, 256, 256, 3], dtype=tf.float32)  # Shape: [10, 256, 256, 3]
labels = tf.range(10, dtype=tf.int32)

# Combine into a dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

def augment_fn(x):
    img = x[0]  # Original image tensor (float32)
    # Incorrectly casting the image to int64 instead of float32
    img = tf.cast(img, dtype=tf.int64) 
    return img

# Apply the augment function to process each sample
augmented_dataset = dataset.map(augment_fn)

# The issue arises when attempting operations expecting compatible dtypes

def correct_augment_fn(x):
    img = x[0]
    # Correctly keep image as float32 for proper preprocessing
    img = tf.cast(img, dtype=tf.float32)
    return img