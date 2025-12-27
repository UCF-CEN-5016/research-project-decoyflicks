!pip install tensorflow-model-garden
import tensorflow as tf
from object_detection.utils import colab_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

# This code should reproduce the error
model = model_builder.build(model_config_path='path/to/model.config')