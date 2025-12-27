import tensorflow as tf
from tensorflow.keras import layers, ops

# Step 1: Generate a synthetic image (int32, e.g., from a dataset)
image = tf.random.uniform(shape=(72, 72, 3), dtype=tf.int32)  # This will cause the error

# Step 2: Convert the image to float32 and normalize
image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

# Step 3: Resize the image (optional, depends on your use case)
resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), 
    size=(72, 72)  # Ensure the dimensions match your input
)

# Step 4: Apply the Patches layer
patch_size = 8
patches_layer = layers.Patches(patch_size=patch_size)
result = patches_layer(resized_image)

print("Shape of patches:", result.shape)