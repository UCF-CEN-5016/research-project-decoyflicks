import matplotlib.pyplot as plt

# Plot works normally before importing tensorflow_models
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Plot before importing tensorflow_models")
plt.show()

# Import tensorflow_models which internally sets matplotlib backend to 'agg'
import tensorflow_models  # This will switch matplotlib backend to 'agg'

# Check backend after import
print("Backend after importing tensorflow_models:", plt.get_backend())

# Plot does not show because the backend is non-interactive
plt.plot([1, 2, 3], [6, 5, 4])
plt.title("Plot after importing tensorflow_models")
plt.show()