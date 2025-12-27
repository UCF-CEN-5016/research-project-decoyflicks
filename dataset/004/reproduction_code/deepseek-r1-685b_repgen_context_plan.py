import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def beta_sample(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, alpha, beta=1.0)
    sample_beta = tf.random.gamma(shape, beta, beta=1.0)
    return sample_alpha / (sample_alpha + sample_beta)

# Compare distributions
n_samples = 100000
alpha, beta = 0.2, 0.2

# Generate samples
correct = beta_sample(alpha, beta, [n_samples]).numpy()
reference = np.random.beta(alpha, beta, n_samples)

# Plot comparison
plt.figure(figsize=(10, 6))
sns.kdeplot(correct, clip=(0, 1), label='TensorFlow Implementation')
sns.kdeplot(reference, clip=(0, 1), label='Numpy Reference')
plt.title('Beta(0.2, 0.2) Sampling Comparison')
plt.legend()
plt.show()