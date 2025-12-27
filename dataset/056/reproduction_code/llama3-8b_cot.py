import tensorflow as tf
from tensorflow import keras
import numpy as np

image_size = 72
patch_size = 14
x_train = ...  # Replace with your image data

resized_image = tf.image.resize(x_train[np.random.choice(range(x_train.shape[0]))], (image_size, image_size))
patches = keras.preprocessing.image.Patches(patch_size)(resized_image)

import tensorflow as tf
from tensorflow import keras
import numpy as np

image_size = 72
patch_size = 14
x_train = ...  # Replace with your image data

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(x_train[np.random.choice(range(x_train.shape[0]))], (image_size, image_size))
patches = keras.preprocessing.image.Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(tf.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")