"""Refactored access to the model_builder from object_detection.builders.

This module re-exports the original `model_builder` and also provides
a small accessor function and a canonical constant name for clarity.
No core logic or functionality is changed.
"""

from object_detection.builders import model_builder as _model_builder

# Keep original name for compatibility
model_builder = _model_builder

# Canonical constant name for clearer use sites
MODEL_BUILDER = _model_builder

def get_model_builder():
    """Return the imported model_builder object.

    This accessor preserves the original imported object while providing
    a clear entry point for callers who prefer function-style access.
    """
    return _model_builder

__all__ = ["model_builder", "MODEL_BUILDER", "get_model_builder"]