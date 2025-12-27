import tensorflow as tf

# Mixed data types
images = tf.cast(tf.random.normal((1, 160, 160, 3)), tf.float32)
segmentation_masks = tf.random.uniform((1, 160, 160, 1), minval=0, maxval=255, dtype=tf.int64)

# RandAugment layer with default augmentation
rand_augment_layer = tf.keras.layers.RandAugment(
    height_shift_max=0.2,
    width_shift_max=0.2,
    brightness_max_delta=0.5,
    contrast_max_factor=1.8,
    saturation_max_factor=1.8,
    hue_max_delta=0.1
)

# Apply augmentation to inputs
augmented_inputs = rand_augment_layer({'images': images, 'segmentation_masks': segmentation_masks})

print("Augmented Inputs:", augmented_inputs)