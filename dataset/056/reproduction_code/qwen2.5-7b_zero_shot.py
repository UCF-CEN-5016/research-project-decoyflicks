import tensorflow as tf
from tensorflow.keras import layers, activations

def generate_synthetic_image(shape=(72, 72, 3)):
    image = tf.random.uniform(shape=shape, dtype=tf.int32)
    return tf.cast(image, tf.float32) / 255.0

def resize_image(image, size=(72, 72)):
    return tf.image.resize(image, size)

def apply_patches_layer(image, patch_size=8):
    patches_layer = layers.Patches(patch_size=patch_size)
    return patches_layer(image)

# Step 1: Generate a synthetic image
image = generate_synthetic_image()

# Step 2: Resize the image
resized_image = resize_image(image)

# Step 3: Apply the Patches layer
result = apply_patches_layer(resized_image)

print("Shape of patches:", result.shape)