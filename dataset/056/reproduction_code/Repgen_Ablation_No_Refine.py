import numpy as np
from keras import layers
from tensorflow.keras import ops
from keras.datasets import cifar10

# Set image size and patch size
image_size = 72
patch_size = 16

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Create a random image from the training set and convert it to int32 tensor
image_index = np.random.choice(range(x_train.shape[0]))
image_tensor = np.expand_dims(x_train[image_index], axis=0).astype(np.int32)

# Resize the image
resized_image = ops.image.resize(
    image_tensor, size=(image_size, image_size), method="nearest"
)

# Verify resized image shape and dtype
assert resized_image.shape == (1, 72, 72, 3)
assert resized_image.dtype == np.int32

# Create a Patches layer with patch_size=16
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = ops.image.extract_patches(
            images, size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size)
        )
        patches = tf.reshape(patches, (batch_size, num_patches_h * num_patches_w, -1))
        return patches

# Call the Patches layer on the resized image
patches = Patches(patch_size)(resized_image)

# Verify that an InvalidArgumentError occurs during the call to Patches.layer()