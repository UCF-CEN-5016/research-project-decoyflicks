import tensorflow as tf
from keras_cv import layers

# Assuming train_ds is defined elsewhere, e.g., from the example
train_ds = ...  # Original dataset pipeline for training

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda images, segmentation_masks: (tf.cast(images, tf.float32), tf.cast(segmentation_masks, tf.float32)))
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# For validation dataset
resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda images, segmentation_masks: (tf.cast(images, tf.float32), tf.cast(segmentation_masks, tf.float32)))
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define augment_fn ensuring all tensors are float32
def augment_fn(image, segmentation_mask):
    # Ensure image is cast to float32 for augmentation
    image = tf.cast(image, tf.float32) / 255.0
    
    # Cast segmentation mask if it's not already float32 (assuming it's int64)
    segmentation_mask = tf.cast(segmentation_mask, tf.float32) / 255.0
    
    # Apply any other augmentations here
    return image, segmentation_mask

.map(lambda images, segmentation_masks: (tf.cast(images, tf.float32), tf.cast(segmentation_masks, tf.float32)))