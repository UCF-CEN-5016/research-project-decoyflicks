import tensorflow as tf
import os

# Minimal setup: a list of image file paths, including one that does not exist
image_paths = [
    'existing_image_1.jpg',  # Assume this exists
    'non_existent_image.jpg' # This file does NOT exist
]

# For testing, create a dummy image file for 'existing_image_1.jpg'
import numpy as np
from PIL import Image

if not os.path.exists('existing_image_1.jpg'):
    dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    Image.fromarray(dummy_image).save('existing_image_1.jpg')

# Function to load images from path
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # normalize
    return image

# Create a dataset from the file paths
path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
image_ds = path_ds.map(load_image)

# Create dummy labels
labels = [0, 1]
label_ds = tf.data.Dataset.from_tensor_slices(labels)

# Combine into a dataset of (image, label)
dataset = tf.data.Dataset.zip((image_ds, label_ds))
dataset = dataset.batch(2)

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# This will raise NotFoundError when reading 'non_existent_image.jpg'
model.fit(dataset, epochs=1)