import matplotlib
import matplotlib.pyplot as plt
import tensorflow_models as tfm

# Before importing tensorflow_models - works
print("Backend before tfm:", matplotlib.get_backend())
plt.plot([1, 2, 3])
plt.title("Plot before tfm import - shows")
plt.show()

# After importing tensorflow_models - fails
print("Backend after tfm:", matplotlib.get_backend())
plt.plot([3, 2, 1])
plt.title("Plot after tfm import - doesn't show")
plt.show()

# Workaround: Set backend before importing tfm
matplotlib.use('TkAgg')  # Must be before any other matplotlib imports
import matplotlib.pyplot as plt
import tensorflow_models as tfm

print("Backend with workaround:", matplotlib.get_backend())
plt.plot([1, 3, 2])
plt.title("Plot with workaround - shows")
plt.show()