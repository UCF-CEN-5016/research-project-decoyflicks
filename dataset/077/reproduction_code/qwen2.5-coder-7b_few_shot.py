import tensorflow as tf
from typing import List

IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
LOG_DIR = './logs'
FILE_PATHS: List[str] = [
    'drive/MyDrive/Dog Vision/train/17e00d79ad69729522d8705e95939f01.jpg'
]


def build_model() -> tf.keras.Model:
    """Constructs and returns the Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMAGE_SHAPE),
        tf.keras.layers.Dense(10)
    ])
    return model


def preprocess_image(file_path: tf.Tensor) -> tf.Tensor:
    """Reads an image file, decodes, resizes, and normalizes it."""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def prepare_dataset(file_paths: List[str]) -> tf.data.Dataset:
    """Creates a tf.data.Dataset of preprocessed images from file paths."""
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(preprocess_image)
    return dataset


def train_model() -> tf.keras.Model:
    """Builds, compiles, and trains the model; returns the trained model."""
    model = build_model()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

    train_dataset = prepare_dataset(FILE_PATHS)
    val_dataset = train_dataset

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[tensorboard_cb])

    return model


model = train_model()