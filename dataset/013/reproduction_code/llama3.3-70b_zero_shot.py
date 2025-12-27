import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

# Before importing tensorflow_models
plt.plot([1, 2, 3])
plt.show()

print("Before importing tensorflow_models, the backend is: ", matplotlib.get_backend())

import tensorflow_models as tfm

# After importing tensorflow_models
print("After importing tensorflow_models, the backend is: ", matplotlib.get_backend())

plt.plot([4, 5, 6])
plt.show()

# Changing to TkAgg, an interactive backend, after importing tensorflow_models
matplotlib.use('TkAgg')
plt.plot([7, 8, 9])
plt.show()

# Changing to the backend before importing tensorflow_models
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_models as tfm

plt.plot([10, 11, 12])
plt.show()