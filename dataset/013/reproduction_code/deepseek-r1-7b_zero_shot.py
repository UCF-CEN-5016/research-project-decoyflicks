import tensorflow as tf
import tensorflow_models as tfm

# Set matplotlib backend
import matplotlib.pyplot as plt
plt.figure()  # Create a new figure
plt.aggSHOW = True  # Ensure backend supports showing plots

x = range(10)
y = [i**2 for i in x]

plt.plot(x, y)
plt.show()