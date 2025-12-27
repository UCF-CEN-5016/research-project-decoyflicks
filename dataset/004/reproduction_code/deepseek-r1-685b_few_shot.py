import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def incorrect_beta_sample(alpha, beta, shape):
    """Incorrect implementation from tfm.vision.augment"""
    sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
    sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
    return sample_alpha / (sample_alpha + sample_beta)

def correct_beta_sample(alpha, beta, shape):
    """Correct beta distribution sampling"""
    sample_alpha = tf.random.gamma(shape, alpha, beta=1.0)
    sample_beta = tf.random.gamma(shape, beta, beta=1.0)
    return sample_alpha / (sample_alpha + sample_beta)

# Compare distributions
n_samples = 100000
alpha, beta = 0.2, 0.2  # Common MixUp parameters

# Generate samples
incorrect = incorrect_beta_sample(alpha, beta, [n_samples]).numpy()
correct = correct_beta_sample(alpha, beta, [n_samples]).numpy()
reference = np.random.beta(alpha, beta, n_samples)  # Numpy's correct implementation

# Plot comparison
plt.figure(figsize=(10, 6))
sns.kdeplot(incorrect, clip=(0, 1), label='Incorrect TF Implementation')
sns.kdeplot(correct, clip=(0, 1), label='Correct Implementation')
sns.kdeplot(reference, clip=(0, 1), label='Numpy Reference')
plt.title('Beta(0.2, 0.2) Sampling Comparison')
plt.legend()
plt.show()