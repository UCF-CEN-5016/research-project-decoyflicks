import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MixupAndCutmix:
    @staticmethod
    def _sample_from_beta_wrong(alpha, beta, shape):
        # INCORRECT: swaps alpha and beta as shape and scale in gamma sampling
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    @staticmethod
    def _sample_from_beta_correct(alpha, beta, shape):
        # CORRECT: alpha and beta as shape parameters, scale fixed to 1.0
        sample_alpha = tf.random.gamma(shape, alpha, beta=1.0)
        sample_beta = tf.random.gamma(shape, beta, beta=1.0)
        return sample_alpha / (sample_alpha + sample_beta)

shape = [10000]

# Sample using incorrect method
samples_wrong = MixupAndCutmix._sample_from_beta_wrong(0.2, 0.2, shape).numpy()

# Sample from true numpy beta distribution for reference
samples_true = np.random.beta(0.2, 0.2, shape[0])

# Sample using correct method
samples_correct = MixupAndCutmix._sample_from_beta_correct(0.2, 0.2, shape).numpy()

# Plot KDEs for comparison
sns.kdeplot(samples_wrong, clip=(0, 1), label='Wrong tf gamma sampling')
sns.kdeplot(samples_correct, clip=(0, 1), label='Correct tf gamma sampling')
sns.kdeplot(samples_true, clip=(0, 1), label='Numpy beta distribution')
plt.legend()
plt.title("Incorrect vs Correct Beta Sampling in MixupAndCutmix")
plt.show()