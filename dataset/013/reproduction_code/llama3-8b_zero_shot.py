import matplotlib.pyplot as plt
import tensorflow as tf

# This line is key: setting the backend before importing tensorflow
plt.switch_backend('TkAgg')

plt.plot([1, 2, 3])
plt.show()

import tensorflow_models

plt.plot([4, 5, 6])
plt.show()