# Install dependencies
# pip install tensorflow==2.13.10 matplotlib tf-models-nightly

import tensorflow as tf
import matplotlib.pyplot as plt
from official.vision import model as tf_models

# Set the Matplotlib backend to 'Agg'
plt.switch_backend('Agg')

# Create a simple plot
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()

# Import TensorFlow Model Garden
from official.vision import model as tf_models

# Create another simple plot
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()

# Check the current Matplotlib backend
print(plt.get_backend())

# Attempt to switch the backend to 'TkAgg'
plt.switch_backend('TkAgg')

# Create a plot again
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()