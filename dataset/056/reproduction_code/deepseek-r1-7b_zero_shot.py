import os
import numpy as np
from tensorflow.keras.applications import VisionTransformer
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the training data (replace with your actual data loading)
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

image_size = 72
patch_size = 16

def preprocess_image(image):
    """Convert image to float32 format"""
    return image.astype(np.float32) / 255.0

# Process each image in the training data
processed_images = np.array([preprocess_image(img) for img in x_train])

# Now create the Patches layer and process images
patches_layer = VisionTransformer.patch_embedding.patches
resized_images = [
    ops.image.resize(
        ops.convert_to_tensor([img]), size=(image_size, image_size)
    )
    for img in processed_images
]

# Alternatively, using TensorFlow operations:
import tensorflow as tf

processed_images = tf.stack(processed_images)
resized_images = tf.image.resize(processed_images, (image_size, image_size))

# Create patches
patches = Patches(patch_size)(resized_images)

print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")