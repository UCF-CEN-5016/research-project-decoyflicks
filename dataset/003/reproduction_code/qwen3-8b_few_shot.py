# Minimal reproduction of the AttributeError
import tensorflow as tf
from object_detection.builders import model_builder

# This import will trigger the AttributeError
from object_detection.utils import config_util

print("Imports completed successfully")