import matplotlib.pyplot as plt
import tensorflow_models as tfm

def plot_data(data, title):
    plt.plot(data)
    plt.title(title)
    plt.show()

# Before importing tensorflow_models
print("Backend before tfm:", plt.get_backend())
plot_data([1, 2, 3], "Plot before tfm import - shows")

# After importing tensorflow_models
print("Backend after tfm:", plt.get_backend())
plot_data([3, 2, 1], "Plot after tfm import - doesn't show")

# Workaround: Set backend before importing tfm
plt.switch_backend('TkAgg')  # Must be before any other matplotlib imports
import matplotlib.pyplot as plt
import tensorflow_models as tfm

print("Backend with workaround:", plt.get_backend())
plot_data([1, 3, 2], "Plot with workaround - shows")