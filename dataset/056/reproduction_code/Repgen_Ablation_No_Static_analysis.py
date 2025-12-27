import tensorflow as tf
from keras import layers

# Set the image size and patch size
image_size = 72
patch_size = 16

# Load and preprocess a sample dataset (e.g., CIFAR-10)
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)

# Convert x_train to float32 type to simulate the issue
x_train = tf.cast(x_train, tf.float32)

# Resize x_train images to the image_size (72x72) with method='nearest'
resized_image = tf.image.resize(x_train, size=(image_size, image_size), method='nearest')

# Call the Patches class constructor with the defined patch size and call it on the resized_image tensor
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        return tf.reshape(patches, [batch_size, -1, self.patch_size * self.patch_size * 3])

patches = Patches(patch_size)(resized_image)

# Verify that an InvalidArgumentError is raised due to the expected input type being int32 but receiving float32
try:
    patches = Patches(patch_size)(resized_image)
except tf.errors.InvalidArgumentError as e:
    print(e.message)