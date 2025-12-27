import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Set to interactive backend before importing

import tensorflow_models  # This may change the matplotlib backend

plt.plot([1, 2, 3])
plt.title("Test Plot")
plt.show()