"""
Module providing a placeholder ExponentialMovingAverage optimizer subclass.

This module preserves the original behavior: defines an optimizer class that
inherits from keras.optimizers.legacy.Optimizer without adding functionality,
and provides a factory to create an instance.
"""

import tensorflow as tf
from tensorflow import keras

__all__ = ["ExponentialMovingAverage", "create_exponential_moving_average", "optimizer"]


class ExponentialMovingAverage(keras.optimizers.legacy.Optimizer):
    """Placeholder optimizer subclass for Exponential Moving Average.

    This class intentionally does not add or override any behavior. It exists
    to provide a recognizable type that subclasses the legacy Keras Optimizer.
    """
    pass


def create_exponential_moving_average() -> ExponentialMovingAverage:
    """Create and return an ExponentialMovingAverage optimizer instance."""
    return ExponentialMovingAverage()


# Instance kept for compatibility with original code.
optimizer = create_exponential_moving_average()