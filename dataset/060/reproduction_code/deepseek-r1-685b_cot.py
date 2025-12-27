import tensorflow as tf
from keras_cv.layers import RandAugment

# Create a minimal dataset with image+mask pairs
images = tf.random.uniform((2, 160, 160, 3), dtype=tf.float32)
masks = tf.random.uniform((2, 160, 160, 1), minval=0, maxval=2, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices({"images": images, "segmentation_masks": masks})

# Create augmentation pipeline
augmenter = RandAugment(
    value_range=(0, 1),
    augmentations_per_image=2,
    magnitude=0.5,
    rate=0.5,
)

def augment_fn(inputs):
    return augmenter(inputs)

# This will trigger the bug
augmented_ds = (
    dataset
    .batch(1)
    .map(augment_fn)
)

# Try to consume the dataset to trigger the error
for batch in augmented_ds:
    print(batch)  # Error occurs here