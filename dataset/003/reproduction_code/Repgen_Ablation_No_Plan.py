import tensorflow as tf
from object_detection.utils import colab_utils, config_util
from object_detection.builders import model_builder

class FreezableSyncBatchNorm(tf.keras.layers.experimental.SyncBatchNormalization):
    def __init__(self, **kwargs):
        super(FreezableSyncBatchNorm, self).__init__(**kwargs)

# Example usage
model = model_builder.build_model()