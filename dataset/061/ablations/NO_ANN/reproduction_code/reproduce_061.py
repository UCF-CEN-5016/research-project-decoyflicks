import os
import tensorflow as tf
import keras
import keras_cv
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "tensorflow"

data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
BATCH_SIZE = 32

def to_dict(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, dataset_info.features["label"].num_classes)
    return {"images": image, "labels": label}

def prepare_dataset(dataset, split):
    if split == "train":
        return dataset.shuffle(10 * BATCH_SIZE).map(to_dict, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    if split == "test":
        return dataset.map(to_dict, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

train_dataset = prepare_dataset(data['train'], 'train')

from keras.utils import data_utils

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)