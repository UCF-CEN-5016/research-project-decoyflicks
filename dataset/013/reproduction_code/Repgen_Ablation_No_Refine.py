# Import necessary libraries
import matplotlib.pyplot as plt

# Verify current matplotlib backend
print(plt.get_backend())

# Change matplotlib backend and verify
plt.switch_backend('TkAgg')
print(plt.get_backend())
plt.switch_backend('Agg')
print(plt.get_backend())
plt.switch_backend('Qt5Agg')
print(plt.get_backend())