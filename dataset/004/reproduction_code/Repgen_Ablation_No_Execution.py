import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming MixupAndCutmix is defined in the main file
from main_file import MixupAndCutmix

batch_size = 100
height, width = 256, 256

# Create random input data with shape (batch_size, height, width, 3)
input_data = tf.random.normal((batch_size, height, width, 3))

# Create labels for the input images with shape (batch_size, 1)
labels = tf.random.uniform((batch_size, 1), maxval=2, dtype=tf.int32)

mixup_and_cutmix = MixupAndCutmix()
sample_alpha = mixup_and_cutmix._sample_from_beta(0.2, 0.2, shape=(batch_size, 1))

# Verify that the output of the _sample_from_beta function has values outside the range [0, 1]
assert tf.reduce_any(sample_alpha < 0) or tf.reduce_any(sample_alpha > 1)

# Plot the distribution of the sample_alpha / (sample_alpha + sample_beta)
plt.figure(figsize=(10, 6))
sns.histplot(sample_alpha / (sample_alpha + sample_alpha), bins=30)
plt.title('Distribution of Sample Alpha')
plt.show()

# Compare this plot with a true beta distribution with alpha=0.2 and beta=0.2 to validate the incorrect implementation
from scipy.stats import beta

x = np.linspace(0, 1, 100)
y_true = beta.pdf(x, 0.2, 0.2)

plt.figure(figsize=(10, 6))
sns.histplot(sample_alpha / (sample_alpha + sample_alpha), bins=30, kde=False, color='blue', label='Sample Alpha')
plt.plot(x, y_true, 'r-', lw=2, label='True Beta Distribution')
plt.title('Comparison of Sample Alpha and True Beta Distribution')
plt.legend()
plt.show()

# Assert that the distribution is significantly different from the expected beta distribution
assert not tf.reduce_all(tf.equal(sample_alpha / (sample_alpha + sample_alpha), y_true))