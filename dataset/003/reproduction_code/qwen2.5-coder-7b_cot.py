from object_detection.core import model_builder as MODEL_BUILDER

import tensorflow as tf
from typing import Dict, Type

# Provide SyncBatchNormalization when available; otherwise fall back to BatchNormalization
SyncBatchNormalization = getattr(tf.keras.layers, "SyncBatchNormalization", tf.keras.layers.BatchNormalization)

# Keras layer aliases
Layer = tf.keras.layers.Layer
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dropout = tf.keras.layers.Dropout

# Export a convenient mapping of layer names to classes for external use
LAYERS: Dict[str, Type[Layer]] = {
    "SyncBatchNormalization": SyncBatchNormalization,
    "Layer": Layer,
    "Input": Input,
    "Dense": Dense,
    "Flatten": Flatten,
    "Conv2D": Conv2D,
    "MaxPool2D": MaxPool2D,
    "GlobalAveragePooling2D": GlobalAveragePooling2D,
    "Dropout": Dropout,
}

__all__ = [
    "MODEL_BUILDER",
    "SyncBatchNormalization",
    "Layer",
    "Input",
    "Dense",
    "Flatten",
    "Conv2D",
    "MaxPool2D",
    "GlobalAveragePooling2D",
    "Dropout",
    "LAYERS",
]