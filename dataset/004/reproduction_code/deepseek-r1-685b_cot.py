import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from official.vision.ops import augment

# Correct beta sampling using gamma distributions
def correct_beta_sample(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, alpha, beta=1.0)
    sample_beta = tf.random.gamma(shape, beta, beta=1.0)
    return sample_alpha / (sample_alpha + sample_beta)

# Parameters for beta distribution
alpha, beta = 0.2, 0.2
num_samples = 100000

# Generate samples from both implementations
buggy_samples = augment.MixupAndCutmix._sample_from_beta(alpha, beta, [num_samples]).numpy()
correct_samples_tf = correct_beta_sample(alpha, beta, [num_samples]).numpy()
correct_samples_np = np.random.beta(alpha, beta, num_samples)

# Plot the distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(buggy_samples, clip=(0, 1), label='Buggy TF Implementation')
sns.kdeplot(correct_samples_tf, clip=(0, 1), label='Correct TF Implementation')
sns.kdeplot(correct_samples_np, clip=(0, 1), label='NumPy Beta Sampling')
plt.title('Comparison of Beta Distribution Sampling')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()