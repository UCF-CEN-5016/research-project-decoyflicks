# repro.py
import tensorflow as tf

# Attempt to import from tensorflow.contrib (which is no longer available in TF 2.x)
from tensorflow.contrib.quantize.python import graph_matcher

print("Successfully imported graph_matcher from tensorflow.contrib")