import os
import numpy as np
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from keras.utils import load_img
from PIL import ImageOps

BATCH_SIZE = 32
img_size = (160, 160)
input_dir = 'images/'
target_dir = 'annotations/trimaps/'

input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith('.jpg')])
target_img_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith('.png') and not fname.startswith('.')])

def load_img_masks(input_img_path, target_img_path):
    input_img = tf_io.read_file(input_img_path)
    input_img = tf_io.decode_png(input_img, channels=3)
    input_img = tf_image.resize(input_img, img_size)
    input_img = tf_image.convert_image_dtype(input_img, 'float32')

    target_img = tf_io.read_file(target_img_path)
    target_img = tf_io.decode_png(target_img, channels=1)
    target_img = tf_image.resize(target_img, img_size, method='nearest')
    target_img = tf_image.convert_image_dtype(target_img, 'uint8')
    target_img -= 1
    return input_img, target_img

train_dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
train_dataset = train_dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE).batch(BATCH_SIZE)

def augment_fn(inputs):
    return inputs

def unpackage_inputs(inputs):
    return inputs

augmented_train_ds = train_dataset.shuffle(BATCH_SIZE * 2).map(augment_fn, num_parallel_calls=tf_data.AUTOTUNE).batch(BATCH_SIZE).map(unpackage_inputs).prefetch(buffer_size=tf.data.AUTOTUNE)