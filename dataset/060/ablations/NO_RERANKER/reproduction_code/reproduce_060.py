import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import visualization
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
AUTOTUNE = tf.data.AUTOTUNE

image_paths = tf.ragged.constant(["/path/to/image1.jpg", "/path/to/image2.jpg"])
bbox = tf.ragged.constant([[[0.0, 0.0, 1.0, 1.0]]])
classes = tf.ragged.constant([[0]])

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

num_val = int(len(image_paths) * SPLIT_RATIO)
val_data = data.take(num_val)
train_data = data.skip(num_val)

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_dataset(image_path, classes, bbox):
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

augmenter = keras.Sequential([
    keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
    keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"),
    keras_cv.layers.JitteredResize(target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"),
])

train_ds = train_data.map(load_dataset, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=AUTOTUNE)

resizing = keras_cv.layers.JitteredResize(target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy")
val_ds = val_data.map(load_dataset, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=AUTOTUNE)

def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]

train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=AUTOTUNE)

for images, bounding_boxes in train_ds.take(1):
    print(images.shape, bounding_boxes)

for images, bounding_boxes in val_ds.take(1):
    print(images.shape, bounding_boxes)

try:
    augmented_train_ds = train_ds
    resized_val_ds = val_ds
except Exception as e:
    print(e)