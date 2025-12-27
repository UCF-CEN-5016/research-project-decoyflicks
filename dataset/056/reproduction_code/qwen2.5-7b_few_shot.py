import tensorflow as tf

def resize_image(image, image_size):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    resized_image = tf.image.resize(image, size=(image_size, image_size))
    return resized_image

# Assuming image is a numpy array of type uint8
image = x_train[np.random.choice(x_train.shape[0])]
image_size = 224  # Example image size
resized_image = resize_image(image, image_size)

# Now pass resized_image to the Patches layer
patches = Patches(patch_size)(resized_image)