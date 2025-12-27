import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, utils  # Fixed import for keras.utils
from medmnist import INFO

np.random.seed(42)
tf.random.set_seed(42)

def download_and_prepare_dataset(data_info: dict):
    data_path = utils.get_file(origin=data_info["url"], md5_hash=data_info["MD5"])  # Updated to use utils
    with np.load(data_path) as data:
        train_videos = data["train_images"]
        return train_videos

DATASET_NAME = "organmnist3d"
prepared_dataset = download_and_prepare_dataset(INFO[DATASET_NAME])
x_train = prepared_dataset[0]
image_size = 72
patch_size = 16

# Select a random image from the training set
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.figure(figsize=(4, 4))
plt.imshow(image.astype("uint8"))
plt.axis("off")

# Resize the image and convert it to a tensor
resized_image = tf.image.resize(tf.convert_to_tensor([image]), size=(image_size, image_size))  # Fixed to use tf.image.resize
# Extract patches from the resized image
patches = layers.Patches(patch_size)(resized_image)

# Print the dimensions and patch information
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")