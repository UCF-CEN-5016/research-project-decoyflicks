import matplotlib.pyplot as plt

# Plot works fine before importing tensorflow_models
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Plot before tensorflow_models import")
plt.show()

# Import tensorflow_models (this triggers matplotlib backend change)
import tensorflow_models  # or from tf_model import vision, depending on your install

# Check backend after import
import matplotlib
print(f"Matplotlib backend after importing tensorflow_models: {matplotlib.get_backend()}")

# Plot again - this will not show interactively if backend switched to 'agg'
plt.plot([1, 2, 3], [6, 5, 4])
plt.title("Plot after tensorflow_models import")
plt.show()