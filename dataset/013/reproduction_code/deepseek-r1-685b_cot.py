import matplotlib
import matplotlib.pyplot as plt

# Initial state - should be interactive backend
print("Before tensorflow_models import:")
print(f"Current backend: {matplotlib.get_backend()}")
plt.plot([1, 2, 3])  # Should show plot
plt.show()

# Now import tensorflow_models which changes the backend
from tensorflow_models import vision

# Post-import state
print("\nAfter tensorflow_models import:")
print(f"Current backend: {matplotlib.get_backend()}")
plt.plot([1, 2, 3])  # Won't show plot
plt.show()

# Attempt to change backend (won't work)
matplotlib.use('TkAgg', force=True)
print("\nAfter attempted backend change:")
print(f"Current backend: {matplotlib.get_backend()}")
plt.plot([1, 2, 3])  # Still won't show plot
plt.show()

import matplotlib
# MUST set backend BEFORE any other imports
matplotlib.use('TkAgg')  # or other interactive backend

# Now proceed with imports
import matplotlib.pyplot as plt
from tensorflow_models import vision

# Will work correctly
plt.plot([1, 2, 3])
plt.show()