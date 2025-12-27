import numpy as np
import keras
from keras import ops
from keras.layers import Layer

class Patches(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = ops.shape(images)[0]
        height = ops.shape(images)[1]
        width = ops.shape(images)[2]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * 3,
            ),
        )
        return patches

image_size = 72
patch_size = 6
x_train = np.random.randint(0, 256, size=(1, 32, 32, 3), dtype=np.uint8)
image = x_train[0]
resized_image = ops.image.resize(ops.convert_to_tensor([image]), size=(image_size, image_size)
patches = Patches(patch_size)(resized_image)