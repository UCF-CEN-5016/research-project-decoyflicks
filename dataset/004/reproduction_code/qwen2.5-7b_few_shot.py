import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sample_from_beta(alpha, beta, shape):
    sample_alpha = np.random.gamma(alpha, 1.0, shape)
    sample_beta = np.random.gamma(beta, 1.0, shape)
    return sample_alpha / (sample_alpha + sample_beta)

# Generate samples using numpy's beta function
shape = 1000
samples_correct = np.random.beta(0.2, 0.2, shape)

# Generate samples using the custom function
samples_incorrect = sample_from_beta(0.2, 0.2, shape)

# Plot KDE comparison
plt.figure(figsize=(10, 6))
sns.kdeplot(samples_incorrect, clip=(0, 1), label='Incorrect (Custom Implementation)')
sns.kdeplot(samples_correct, clip=(0, 1), label='Correct (Numpy)')
plt.legend()
plt.title('Beta Distribution Samples (Incorrect vs Correct)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()