import tensorflow as tf

# External placeholders (defined elsewhere in the actual project)
train_ds = ...            # Original dataset pipeline for training
val_ds = ...              # Original dataset pipeline for validation
resize_fn = ...           # Resize mapping function for validation
unpackage_inputs = ...    # Function to unpackage/model-specific input formatting
BATCH_SIZE = ...          # Batch size to be used
AUTOTUNE = tf.data.AUTOTUNE

def cast_pair_to_float32(images, segmentation_masks):
    """Cast a pair of (images, segmentation_masks) to float32."""
    return tf.cast(images, tf.float32), tf.cast(segmentation_masks, tf.float32)

def augment_fn(image, segmentation_mask):
    """Apply augmentation-related casting and normalization."""
    image = tf.cast(image, tf.float32) / 255.0
    segmentation_mask = tf.cast(segmentation_mask, tf.float32) / 255.0
    return image, segmentation_mask

# Training dataset pipeline with augmentation
augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(cast_pair_to_float32)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)

# Validation dataset pipeline with resizing
resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(cast_pair_to_float32)
    .map(unpackage_inputs)
    .prefetch(buffer_size=AUTOTUNE)
)