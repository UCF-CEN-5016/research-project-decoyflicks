import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sample_from_beta(alpha, beta, shape):
    # Incorrect implementation (swapped parameters)
    sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)  # Should be alpha, beta=1.0
    sample_beta = tf.random.gamma(shape, 1.0, beta=beta)    # Should be beta, beta=1.0
    return sample_alpha / (sample_alpha + sample_beta)

# Generate samples using incorrect implementation
shape = tf.shape(tf.range(1000))
samples_incorrect = sample_from_beta(0.2, 0.2, shape).numpy()

# Generate correct samples using numpy's beta function
samples_correct = np.random.beta(0.2, 0.2, 100000)

# Plot KDE comparison
plt.figure(figsize=(10, 6))
sns.kdeplot(samples_incorrect, clip=(0, 1), label='Incorrect (TF Implementation)')
sns.kdeplot(samples_correct, clip=(0, 1), label='Correct (Numpy)')
plt.legend()
plt.title('Beta Distribution Samples (Incorrect vs Correct)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()