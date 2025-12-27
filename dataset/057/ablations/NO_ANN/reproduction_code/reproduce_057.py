import os
import numpy as np
import keras
from keras import layers

os.environ['KERAS_BACKEND'] = 'tensorflow'

try:
    from keras import ops
except ImportError as e:
    print(e)

print("Python version:", os.popen('python --version').read().strip())
print("Keras version:", keras.__version__)
print("OS platform:", os.popen('lsb_release -a').read().strip())
print("GPU model and memory:", os.popen('nvidia-smi').read().strip())