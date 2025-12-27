import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, ops
from keras.layers import Patches

np.random.seed(42)
tf.random.set_seed(42)

image_size = 72
patch_size = 16

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
image = x_train[np.random.choice(range(x_train.shape[0]))]

plt.imshow(image.astype('uint8'))
plt.axis('off')

resized_image = ops.image.resize(ops.convert_to_tensor([image]), size=(image_size, image_size))

patches = Patches(patch_size)(resized_image)

print(f"Image size: {image.shape}, Patch size: {patch_size}, Patches per image: {patches.shape[1]}, Elements per patch: {patches.shape[2]}")

n = int(np.sqrt(patches.shape[1]))
for i, patch in enumerate(patches[0]):
    plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype('uint8'))
    plt.axis('off')

plt.show()