import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sample_from_beta(alpha, beta, size):
    sample = np.random.beta(alpha, beta, size)
    return sample

# Generate samples using the correct implementation
correct_samples = sample_from_beta(0.2, 0.2, 100000)

# Generate samples using TensorFlow
tf_samples = np.random.beta(0.2, 0.2, 100000)

# Plot the distributions
sns.kdeplot(tf_samples, clip=(0, 1), label='TensorFlow')
sns.kdeplot(correct_samples, clip=(0, 1), label='Numpy')
plt.title('Comparison of Beta Distribution Sampling')
plt.legend()
plt.show()