import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure TensorFlow backend

# This line will raise ImportError: cannot import name 'ops' from 'keras'
from keras import ops

print("This line will never execute due to the ImportError")