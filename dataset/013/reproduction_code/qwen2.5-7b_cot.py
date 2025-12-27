import matplotlib.pyplot as plt
import matplotlib

# Step 1: Check the backend before importing tensorflow_models
print("Backend before importing tensorflow_models:", matplotlib.get_backend())

# Step 2: Attempt to plot
plt.plot([1, 2, 3])
plt.title("Plot before importing tensorflow_models")
plt.show()

# Step 3: Change the backend
matplotlib.use('TkAgg')  # or 'Agg' for non-interactive

# Step 4: Re-check backend after changing
print("Backend after setting backend:", matplotlib.get_backend())

# Step 5: Import tensorflow_models
import tensorflow_models

# Step 6: Attempt to plot again
plt.plot([1, 2, 3])
plt.title("Plot after importing tensorflow_models")
plt.show()