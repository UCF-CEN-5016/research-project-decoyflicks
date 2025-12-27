import tensorflow as tf
from object_detection.utils import colab_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

try:
    from object_detection.core.freezable_sync_batch_norm import FreezableSyncBatchNorm
except AttributeError as e:
    print(f"Error: {e}")

print("Code should run without error")

