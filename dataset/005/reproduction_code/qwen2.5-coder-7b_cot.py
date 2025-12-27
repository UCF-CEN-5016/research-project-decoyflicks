# First, deactivate any conflicting virtual environments if necessary
# Install required packages in an activated virtual environment

import tensorflow as tf
from tensorflow.keras import layers, models, utils

__all__ = ["tf", "layers", "models", "utils", "get_keras_components"]


def get_keras_components():
    """Return the commonly used TensorFlow/Keras components from this module."""
    return tf, layers, models, utils


if __name__ == "__main__":
    # Quick sanity check when executed as a script
    _tf, _layers, _models, _utils = get_keras_components()
    print("TensorFlow version:", _tf.__version__)