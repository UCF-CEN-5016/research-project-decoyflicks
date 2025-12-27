import tensorflow as tf
from tensorflow import keras

def resize_image(image, image_size):
    # Use Keras ops to mimic the environment that produced the reported bug:
    # convert the image to a tensor (keeps an integer dtype) and resize with keras.ops.image.resize
    image_tensor = keras.ops.convert_to_tensor([image])
    resized_image = keras.ops.image.resize(image_tensor, size=(image_size, image_size))
    return resized_image

# Assuming image is a numpy array (e.g., from x_train)
image = x_train[np.random.choice(x_train.shape[0])]
image_size = 224  # Example image size
resized_image = resize_image(image, image_size)

# Now pass resized_image to the Patches layer (as in the failing notebook)
patches = Patches(patch_size)(resized_image)