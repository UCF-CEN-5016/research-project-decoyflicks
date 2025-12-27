import tensorflow as tf
import keras_cv

# Create dummy inputs with inconsistent mask dtype
image = tf.random.uniform(shape=(160, 160, 3), dtype=tf.float32)
# segmentation mask with int64 dtype (issue source)
mask = tf.random.uniform(shape=(160, 160, 1), maxval=2, dtype=tf.int64)

inputs = {
    "images": image,
    "segmentation_masks": mask,
}

# Define RandAugment layer from keras_cv with default parameters
rand_augment = keras_cv.layers.RandAugment()

# Call augmentation - this will raise the TypeError
output = rand_augment(inputs)