import os
import tensorflow as tf
from tqdm.auto import tqdm
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

BATCH_SIZE = 4
AUTOTUNE = tf.data.AUTOTUNE

def load_image(image_path):
    return tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)

def load_dataset(image_path, classes, bbox):
    return {
        "images": load_image(image_path),
        "bounding_boxes": {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": bbox,
        }
    }

image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
bbox = [[[0, 0, 1, 1]]]
classes = [[0]]

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
train_data = data.take(1)
val_data = data.skip(1)

def augment_fn(inputs):
    return inputs

augmented_train_ds = (
    train_data.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
)

resized_val_ds = (
    val_data.map(lambda x, y: load_dataset(x, y[0], y[1]), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
)

# Attempt to run the code to create datasets
augmented_train_ds
resized_val_ds