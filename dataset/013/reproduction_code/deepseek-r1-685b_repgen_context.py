import matplotlib.pyplot as plt
import matplotlib
import tensorflow_models as tfm

def plot_and_show(data, title):
    plt.plot(data)
    plt.title(title)
    plt.show()

# Before importing tensorflow_models - works
print("Backend before tfm:", matplotlib.get_backend())
plot_and_show([1, 2, 3], "Plot before tfm import - shows")

# After importing tensorflow_models - fails
print("Backend after tfm:", matplotlib.get_backend())
plot_and_show([3, 2, 1], "Plot after tfm import - doesn't show")

# Workaround: Set backend before importing tfm
matplotlib.use('TkAgg')  # Must be before any other matplotlib imports
import matplotlib.pyplot as plt
import tensorflow_models as tfm

print("Backend with workaround:", matplotlib.get_backend())
plot_and_show([1, 3, 2], "Plot with workaround - shows")