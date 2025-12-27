import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import ops

class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        patches = ops.image.extract_patches(images, size=self.patch_size)
        return ops.reshape(
            patches,
            (patches.shape[0], -1, self.patch_size * self.patch_size * 3),
        )

image_size = 72
patch_size = 6

x_train = np.random.randint(0, 256, (10, image_size, image_size, 3), dtype=np.uint8)
image = x_train[np.random.choice(range(x_train.shape[0]))]

resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)

patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")