"""
Refactored thin wrapper that ensures the keras_cv.models.weights module is imported
and provides a small accessor API without changing original import semantics.
"""

from importlib import import_module

# Import the target module immediately to preserve original side effects and availability.
weights = import_module("keras_cv.models.weights")


def get_weights_module():
    """Return the imported keras_cv.models.weights module."""
    return weights


__all__ = ["weights", "get_weights_module"]