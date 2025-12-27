import os
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_cv

os.environ["KERAS_BACKEND"] = "tensorflow"

data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def to_dict(image, label):
    image = tf.image.resize(image, (224, 224))
    label = tf.one_hot(label, dataset_info.features["label"].num_classes)
    return {"images": image, "labels": label}

def load_dataset(split="train"):
    dataset = data[split]
    return dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

train_dataset = load_dataset()

from keras.utils import data_utils  # This will raise ImportError