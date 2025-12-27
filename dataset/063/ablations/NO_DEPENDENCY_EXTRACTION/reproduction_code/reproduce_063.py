import os
import numpy as np
import keras
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

def load_img_masks(input_img_path, target_img_path):
    input_img = tf_io.read_file(input_img_path)
    input_img = tf_io.decode_png(input_img, channels=3)
    input_img = tf_image.resize(input_img, img_size)
    input_img = tf_image.convert_image_dtype(input_img, "float32")

    target_img = tf_io.read_file(target_img_path)
    target_img = tf_io.decode_png(target_img, channels=1)
    target_img = tf_image.resize(target_img, img_size, method="nearest")
    target_img = tf_image.convert_image_dtype(target_img, "uint8")
    target_img -= 1
    return input_img, target_img

def augment_fn(input_img):
    # Random augmentations that may lead to type mismatch
    return input_img

train_dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
train_dataset = train_dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=batch_size * 2)
train_dataset = train_dataset.map(lambda x, y: (augment_fn(x), y))
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf_data.AUTOTUNE)

for batch in train_dataset.take(1):
    pass