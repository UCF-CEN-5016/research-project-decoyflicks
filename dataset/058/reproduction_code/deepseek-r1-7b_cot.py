import tensorflow as tf
from keras_cv import models
from keras_cv import utils

# Minimal dataset setup for RetinaNet (batch_size=1)
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_label(label_path):
    labels = tf.io.read_file(label_path)
    labels = tf.reshape(labels, [-1])  # Flatten if necessary
    return labels

# Sample dataset (replace with actual paths and annotations)
train_paths = [
    ('images/train/image_000.jpg', 'annotations/train/box_000.txt'),
    ('images/train/image_001.jpg', 'annotations/train/box_001.txt'),
]

batch_size = 1
ds = tf.data.Dataset.from_tensor_slices((train_paths, train_paths))
ds = ds.map(lambda x: (load_image(x[0]), load_label(x[1])))
ds = ds.take(batch_size)