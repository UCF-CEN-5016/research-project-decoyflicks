# Install dependencies
# pip install tensorflow==2.13.10
# pip install tf-models-nightly
# pip install matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf
from official.vision import image_classification

plt.switch_backend('Agg')

# Initial plot to confirm backend issue
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot')
plt.show()

# Import TensorFlow Model Garden
from official.vision import image_classification

# Second plot to confirm issue persists
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot After Import')
plt.show()

# Check current backend
print(plt.get_backend())

# Attempt to switch backend to 'TkAgg'
plt.switch_backend('TkAgg')

# Final plot to confirm issue persists
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Final Test Plot')
plt.show()