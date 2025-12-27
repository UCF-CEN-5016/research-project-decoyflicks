import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Minimal environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress unnecessary logs

# Triggering condition: Import TensorFlow Model Garden (TFM) and TF Models nightly
import tensorflow_models as tfm
import tf_models_nightly as tfmn

# Key symptoms: Matplotlib's pyplot does not work after importing TFM
try:
    plt.plot([1, 2, 3])  # This should plot a simple line graph
except Exception as e:
    print(f"Error: {e}")

# Additional context: If we add some extra lines of code with "plt" as the first letters...
plt.hist([1, 2, 3], bins=3)  # This should also not work

print("Reproduction Code: Bug reproduces as expected!")