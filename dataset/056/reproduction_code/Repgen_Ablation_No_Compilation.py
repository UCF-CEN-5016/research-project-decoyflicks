import numpy as np
import tensorflow as tf

# Define image size as 72x72 pixels
image_size = 72
patch_size = 16

# Create random uniform input data with shape (1, height, width, channels) where channels = 3
input_data = np.random.uniform(size=(1, 50, 50, 3)).astype(np.float32)

# Convert input data type to int32 using tf.cast
input_data_int32 = tf.cast(input_data, dtype=tf.int32)

# Resize input data to the specified image size using keras.ops.image.resize
resized_image = tf.keras.layers.UpSampling2D(size=(image_size // 50))(input_data_int32)

# Create a Patches layer with the defined patch size
Patches = tf.keras.layers.experimental.preprocessing.Patchify(patch_size=patch_size, flatten=True)

# Call the Patches layer on the resized image tensor
patches = Patches(resized_image)

print(f"Patches shape: {patches.shape}")