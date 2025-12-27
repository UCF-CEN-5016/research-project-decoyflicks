import os

_KERAS_BACKEND_ENV = "KERAS_BACKEND"


def configure_keras_backend(backend: str = "tensorflow") -> None:
    """
    Ensure the KERAS_BACKEND environment variable is set.
    This must be done before importing TensorFlow/Keras.
    """
    os.environ[_KERAS_BACKEND_ENV] = backend


# Set KERAS_BACKEND prior to importing TensorFlow/Keras
configure_keras_backend()

import tensorflow as tf
from tensorflow import keras

__all__ = ["tf", "keras", "configure_keras_backend"]