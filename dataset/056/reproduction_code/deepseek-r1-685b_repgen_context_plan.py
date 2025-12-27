import tensorflow as tf

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        return tf.image.extract_patches(images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')

image_size = 72
patch_size = 6

test_image = tf.random.uniform((1, image_size, image_size, 3), minval=0, maxval=256, dtype=tf.float32)

patches_layer = Patches(patch_size)
patches = patches_layer(test_image)

print("Successfully extracted patches:", patches.shape)