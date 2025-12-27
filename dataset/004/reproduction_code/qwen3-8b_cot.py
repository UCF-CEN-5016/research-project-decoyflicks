import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def incorrect_sample_from_beta(alpha, beta, shape):
    # Incorrect parameter order: gamma(shape, 1.0, beta=alpha)
    sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
    # Incorrect parameter order: gamma(shape, 1.0, beta=beta)
    sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
    return sample_alpha / (sample_alpha + sample_beta)

# Generate samples using the incorrect implementation
incorrect_samples = incorrect_sample_from_beta(0.2, 0.2, tf.shape(tf.range(1000))).numpy()

# Generate samples using the correct numpy beta distribution
correct_samples = np.random.beta(0.2, 0.2, 100000)

# Plot the distributions
sns.kdeplot(incorrect_samples, clip=(0, 1), label='Incorrect (TensorFlow)')
sns.kdeplot(correct_samples, clip=(0, 1), label='Correct (numpy)')
plt.title('Comparison of Beta Distribution Sampling')
plt.legend()
plt.show()