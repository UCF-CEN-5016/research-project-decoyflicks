import matplotlib.pyplot as pyplot
import tensorflow as tensorflow

# This line is key: setting the backend before importing tensorflow
pyplot.switch_backend('TkAgg')


def plot_sequence(sequence, plot_lib=pyplot):
    """Plot a sequence of values and display the figure."""
    plot_lib.plot(sequence)
    plot_lib.show()


# First plot
plot_sequence([1, 2, 3])

import tensorflow_models

# Second plot
plot_sequence([4, 5, 6])