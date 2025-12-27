import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from official import tf_models

# Set up the conda environment and install dependencies
os.system("conda create -n tf_env python=3.10 -y")
os.system("conda activate tf_env")
os.system("pip install tensorflow==2.13.10")
os.system("pip install tf-models-nightly")

# Create a Jupyter Notebook
# Import necessary libraries
plt.switch_backend('Agg')
print(plt.get_backend())

# Import TensorFlow Model Garden
from official import tf_models

# Create a simple plot
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot')
plt.show()

# Switch backend to TkAgg after import
plt.switch_backend('TkAgg')
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot After TkAgg')
plt.show()

# Switch backend to TkAgg before import
plt.switch_backend('TkAgg')
from official import tf_models
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot After TkAgg Before Import')
plt.show()