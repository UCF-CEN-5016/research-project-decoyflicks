import matplotlib
import matplotlib.pyplot as plt
print("Before importing tensorflow_models, backend:", matplotlib.get_backend())
import tensorflow_models
print("After importing tensorflow_models, backend:", matplotlib.get_backend())
plt.plot([1, 2, 3], [4, 5, 1])
plt.show()