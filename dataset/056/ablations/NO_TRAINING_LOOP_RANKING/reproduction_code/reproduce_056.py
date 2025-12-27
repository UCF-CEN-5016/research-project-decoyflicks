import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import img_to_array, load_img
from keras.preprocessing import image_dataset_from_directory

np.random.seed(42)

image_size = 72
patch_size = 16

(x_train, _), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0

random_index = np.random.choice(range(x_train.shape[0]))
selected_image = x_train[random_index]
plt.imshow(selected_image)
plt.axis('off')
plt.show()

resized_image = tf.image.resize(selected_image, [image_size, image_size])
resized_image = tf.convert_to_tensor(resized_image)

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, inputs):
        batch_size, height, width, channels = tf.shape(inputs)
        patches = tf.image.extract_patches(images=inputs,
                                            sizes=[1, self.patch_size, self.patch_size, 1],
                                            strides=[1, self.patch_size, self.patch_size, 1],
                                            rates=[1, 1, 1, 1],
                                            padding='VALID')
        return patches

patches_layer = Patches(patch_size)
patches = patches_layer(tf.cast(resized_image, tf.int32))

print("Image size:", resized_image.shape)
print("Patch size:", patch_size)
print("Patches per image:", patches.shape[1])
print("Elements per patch:", patches.shape[2])

num_patches_per_image = int(np.sqrt(patches.shape[1]))
for i in range(num_patches_per_image):
    plt.imshow(patches[0, i].numpy())
    plt.axis('off')
    plt.show()

assert resized_image.dtype == tf.int32