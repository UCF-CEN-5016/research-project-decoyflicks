import os
import tensorflow as tf
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
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

augmenter = tf.keras.Sequential([
    keras_cv.layers.RandomFlip(mode="horizontal"),
    keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2)
])

train_ds = train_data.map(load_dataset, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(augmenter, num_parallel_calls=AUTOTUNE)

val_ds = val_data.map(load_dataset, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
resizing = keras_cv.layers.JitteredResize(target_size=(640, 640), bounding_box_format="xyxy")
val_ds = val_ds.map(resizing, num_parallel_calls=AUTOTUNE)

augmented_train_ds = train_ds.shuffle(BATCH_SIZE * 2).map(lambda x: x, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).map(lambda x: x, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)