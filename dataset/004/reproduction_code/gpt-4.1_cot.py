import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def incorrect_sample_from_beta(alpha, beta, shape):
    # Incorrect parameter order in tf.random.gamma
    sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)  # WRONG
    sample_beta = tf.random.gamma(shape, 1.0, beta=beta)    # WRONG
    return sample_alpha / (sample_alpha + sample_beta)

def correct_sample_from_beta(alpha, beta, shape):
    # Correct parameter order in tf.random.gamma
    sample_alpha = tf.random.gamma(shape, alpha, beta=1.0)  # CORRECT
    sample_beta = tf.random.gamma(shape, beta, beta=1.0)    # CORRECT
    return sample_alpha / (sample_alpha + sample_beta)

# Parameters
alpha, beta = 0.2, 0.2
num_samples = 100000
shape = [num_samples]

# Sample from incorrect implementation
samples_incorrect = incorrect_sample_from_beta(alpha, beta, shape).numpy()

# Sample from correct implementation
samples_correct = correct_sample_from_beta(alpha, beta, shape).numpy()

# Sample from numpy's beta for ground truth
samples_np = np.random.beta(alpha, beta, num_samples)

# Plotting
sns.kdeplot(samples_incorrect, clip=(0,1), label='Incorrect tf.random.gamma params')
sns.kdeplot(samples_correct, clip=(0,1), label='Correct tf.random.gamma params')
sns.kdeplot(samples_np, clip=(0,1), label='Numpy Beta')
plt.title(f"Sampling Beta({alpha}, {beta}) Distribution")
plt.xlabel("Sample Value")
plt.ylabel("Density")
plt.legend()
plt.show()