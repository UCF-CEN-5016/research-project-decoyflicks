import matplotlib
import matplotlib.pyplot as plt
import tensorflow_models  # This triggers the backend change

# Step 1: Check the backend before importing tensorflow_models
print("Backend before importing tensorflow_models:", matplotlib.get_backend())

# Step 2: Import tensorflow_models (this changes the backend)
# Step 3: Check backend after importing
print("Backend after importing tensorflow_models:", matplotlib.get_backend())

# Step 4: Attempt to plot
plt.plot([1, 2, 3])
plt.title("Plot after importing tensorflow_models")
plt.show()

# 🔧 Fix: Set the backend before importing tensorflow_models
# Run this code before importing tensorflow_models to prevent the backend change
# matplotlib.use('TkAgg')  # or 'Agg' for non-interactive

# Step 5: Re-import tensorflow_models after setting the backend
# import tensorflow_models

# Step 6: Plot again
plt.plot([1, 2, 3])
plt.title("Plot after fixing backend")
plt.show()

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if non-interactive is acceptable
import tensorflow_models