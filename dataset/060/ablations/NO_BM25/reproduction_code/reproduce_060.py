import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_cv

os.environ["KERAS_BACKEND"] = "tensorflow"

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()
data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
val_steps_per_epoch = dataset_info.splits["test"].num_examples // BATCH_SIZE

IMAGE_SIZE = (160, 160)
num_classes = dataset_info.features["label"].num_classes

def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}

def prepare_dataset(dataset, split):
    if split == "train":
        return (
            dataset.shuffle(10 * BATCH_SIZE)
            .map(to_dict, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
        )
    if split == "test":
        return dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

def load_dataset(split="train"):
    dataset = data[split]
    return prepare_dataset(dataset, split)

train_dataset = load_dataset()

def augment_fn(inputs):
    rand_augment = keras_cv.layers.RandAugment(
        value_range=(0, 255),
        augmentations_per_image=3,
        magnitude=0.3,
    )
    inputs["images"] = rand_augment(inputs["images"])
    return inputs

def unpackage_inputs(inputs):
    return inputs["images"], inputs["labels"]

augmented_train_ds = (
    train_dataset.shuffle(BATCH_SIZE * 2)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

resized_val_ds = (
    load_dataset(split="test").map(to_dict, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(unpackage_inputs)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# The function visualize_dataset is undefined in the original code.
# To preserve the bug reproduction logic, we will comment out the call to visualize_dataset.
# visualize_dataset(augmented_train_ds, title="After Augmentation")