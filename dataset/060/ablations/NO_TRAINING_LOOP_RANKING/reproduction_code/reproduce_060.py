import tensorflow as tf
from keras_cv import layers
import numpy as np
import pathlib

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def augment_fn(data):
    images = data['images']
    masks = data['segmentation_masks']
    images = layers.RandomFlip("horizontal")(images)
    images = layers.Rescaling(1.0 / 255)(images)
    return {'images': images, 'segmentation_masks': masks}

def unpackage_inputs(data):
    return data['images'], data['segmentation_masks']

def resize_fn(images, masks):
    images = tf.image.resize(images, (160, 160))
    masks = tf.image.resize(masks, (160, 160))
    return images, masks

url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/oxford_pet_images.zip'
path = tf.keras.utils.get_file('oxford_pet_images.zip', origin=url, extract=True)
data_dir = pathlib.Path(path).parent / 'oxford_pet_images'

train_images = np.load(data_dir / 'train_images.npy')
train_masks = np.load(data_dir / 'train_masks.npy')
val_images = np.load(data_dir / 'val_images.npy')
val_masks = np.load(data_dir / 'val_masks.npy')

train_ds = tf.data.Dataset.from_tensor_slices({'images': train_images, 'segmentation_masks': train_masks})
val_ds = tf.data.Dataset.from_tensor_slices({'images': val_images, 'segmentation_masks': val_masks})

augmented_train_ds = (
    train_ds.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

resized_val_ds = (
    val_ds.map(resize_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Run the code cell that creates 'augmented_train_ds' and 'resized_val_ds'