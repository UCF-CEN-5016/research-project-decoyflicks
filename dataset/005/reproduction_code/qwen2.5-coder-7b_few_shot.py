"""TensorFlow utilities."""

from tensorflow import framework as _tf_framework
import tensorflow as tf

# Expose commonly used names
tensor = _tf_framework.tensor

__all__ = ["tf", "tensor"]