import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()

import tensorflow_models as tfm
plt.plot([4, 5, 6])
plt.show()

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print("Before tfm import:", matplotlib.get_backend())

import tensorflow_models as tfm
print("After tfm import:", matplotlib.get_backend())