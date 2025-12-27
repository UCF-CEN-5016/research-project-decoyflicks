import os


def set_keras_backend_env(backend: str = "tensorflow") -> None:
    """Set the KERAS_BACKEND environment variable before importing Keras."""
    os.environ["KERAS_BACKEND"] = backend


def _import_standard_modules() -> None:
    """Import standard library and common third-party modules into globals."""
    import pathlib as _pathlib
    import random as _random
    import string as _string
    import re as _re
    import numpy as _np

    globals().update(
        {
            "pathlib": _pathlib,
            "random": _random,
            "string": _string,
            "re": _re,
            "np": _np,
        }
    )


def _import_tensorflow_and_keras() -> None:
    """Import TensorFlow and Keras modules into globals, preserving behavior."""
    import tensorflow.data as _tf_data
    import tensorflow.strings as _tf_strings

    import keras as _keras
    from keras import layers as _keras_layers

    try:
        from keras import ops as _keras_ops
    except ImportError:
        _keras_ops = None

    from keras.layers import TextVectorization as _TextVectorization

    globals().update(
        {
            "tf_data": _tf_data,
            "tf_strings": _tf_strings,
            "keras": _keras,
            "layers": _keras_layers,
            "ops": _keras_ops,
            "TextVectorization": _TextVectorization,
        }
    )


# Preserve original script behavior: set env var before importing Keras.
set_keras_backend_env()
_import_standard_modules()
_import_tensorflow_and_keras()