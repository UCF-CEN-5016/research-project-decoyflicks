import matplotlib.pyplot as plt
import tensorflow as tf

# Create a conda environment with python 3.10 and install tensorflow
conda create --name myenv python=3.10 && conda activate myenv && conda install tensorflow

# Import tensorflow models
import tf_models

# Try to plot something
plt.plot([1, 2, 3])
plt.show()