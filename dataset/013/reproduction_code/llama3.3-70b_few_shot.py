import matplotlib
import matplotlib.pyplot as plt
import tensorflow_models as tfm

# Print the initial backend
print("Initial backend:", matplotlib.get_backend())

# Importing tensorflow_models changes the backend to a non-interactive one
import tensorflow_models as tfm
print("Backend after importing tensorflow_models:", matplotlib.get_backend())

# Try to plot something
plt.plot([1, 2, 3])
plt.show()  # This will not display the plot

# Change the backend to an interactive one before importing tensorflow_models
matplotlib.use('TkAgg')
import tensorflow_models as tfm
print("Backend after changing to TkAgg and importing tensorflow_models:", matplotlib.get_backend())

# Now plotting should work
plt.plot([1, 2, 3])
plt.show()  # This will display the plot