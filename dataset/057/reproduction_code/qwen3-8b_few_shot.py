import os

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import ops  # This line triggers the ImportError