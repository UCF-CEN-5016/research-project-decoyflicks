import os
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

def load_img_masks(input_img_path, target_img_path):
    input_img = tf_io.read_file(input_img_path)
    input_img = tf_io.decode_png(input_img, channels=3)
    input_img = tf_image.resize(input_img, (160, 160))
    input_img = tf_image.convert_image_dtype(input_img, "float32")

    target_img = tf_io.read_file(target_img_path)
    target_img = tf_io.decode_png(target_img, channels=1)
    target_img = tf_image.resize(target_img, (160, 160), method="nearest")
    target_img = tf_image.convert_image_dtype(target_img, "int32")

    return input_img, target_img

def get_dataset(batch_size, img_size, input_img_paths, target_img_paths):
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)

batch_size = 32
img_size = (160, 160)
max_dataset_len = 1000

train_input_img_paths = ["path/to/train/images/*.jpg"] * max_dataset_len
train_target_img_paths = ["path/to/train/annotations/*.png"] * max_dataset_len

train_dataset = get_dataset(batch_size, img_size, train_input_img_paths, train_target_img_paths)

val_input_img_paths = ["path/to/val/images/*.jpg"] * 100
val_target_img_paths = ["path/to/val/annotations/*.png"] * 100

val_dataset = get_dataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)