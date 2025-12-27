"""
Module: tensorflow_text loader
Provides the tensorflow_text module alias and a small accessor function.
"""

import tensorflow_text as tfm

__all__ = ["tfm", "get_tensorflow_text"]


def get_tensorflow_text():
    """Return the imported tensorflow_text module (aliased as tfm)."""
    return tfm