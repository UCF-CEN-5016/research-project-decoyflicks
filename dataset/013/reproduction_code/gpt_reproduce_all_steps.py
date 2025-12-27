import matplotlib
import matplotlib.pyplot as plt

print("Backend before importing tensorflow_models:", matplotlib.get_backend())
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

import tensorflow_models

print("Backend after importing tensorflow_models:", matplotlib.get_backend())
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

matplotlib.use('TkAgg')
print("Backend after setting matplotlib.use('TkAgg') post-import:", matplotlib.get_backend())
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()