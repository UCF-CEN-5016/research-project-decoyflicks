import tensorflow as tf

# Assuming image is a numpy array of type uint8
image = x_train[np.random.choice(x_train.shape[0])]
resized_image = tf.image.resize(
    tf.convert_to_tensor(image, dtype=tf.float32), 
    size=(image_size, image_size)
)

# Now pass resized_image to the Patches layer
patches = Patches(patch_size)(resized_image)