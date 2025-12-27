import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Incorrect implementation (current code)
class MixupAndCutmix:
    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha) 
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta) 
        return sample_alpha / (sample_alpha + sample_beta)

# Correct implementation
class CorrectMixupAndCutmix:
    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, alpha, beta=1.0) 
        sample_beta = tf.random.gamma(shape, beta, beta=1.0) 
        return sample_alpha / (sample_alpha + sample_beta)

# Generate incorrect and correct samples
incorrect_samples = MixupAndCutmix._sample_from_beta(0.2, 0.2, tf.shape(tf.range(1000))).numpy()
correct_samples = CorrectMixupAndCutmix._sample_from_beta(0.2, 0.2, tf.shape(tf.range(1000))).numpy()
actual_samples = np.random.beta(0.2, 0.2, 1000)

# Plot the results
sns.kdeplot(incorrect_samples, clip=(0, 1), label='Incorrect Implementation')
sns.kdeplot(correct_samples, clip=(0, 1), label='Correct Implementation')
sns.kdeplot(actual_samples, clip=(0, 1), label='Actual Beta Distribution')
plt.legend()
plt.show()