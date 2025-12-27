import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_cv

os.environ["KERAS_BACKEND"] = "jax"

data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, dataset_info.features["label"].num_classes)
    return {"images": image, "labels": label}

def prepare_dataset(dataset, split):
    if split == "train":
        return (
            dataset.shuffle(10 * BATCH_SIZE)
            .map(to_dict, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
        )
    if split == "test":
        return dataset.map(to_dict, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)

train_dataset = prepare_dataset(data['train'], 'train')

def visualize_dataset(dataset, title):
    plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(dataset.take(9))):
        images = samples["images"]
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

visualize_dataset(train_dataset, title="Before Augmentation")

rand_augment = keras_cv.layers.RandAugment(value_range=(0, 255), augmentations_per_image=3, magnitude=0.3)

def apply_rand_augment(inputs):
    inputs["images"] = rand_augment(inputs["images"])
    return inputs

train_dataset = train_dataset.map(apply_rand_augment)

# This will raise the ImportError