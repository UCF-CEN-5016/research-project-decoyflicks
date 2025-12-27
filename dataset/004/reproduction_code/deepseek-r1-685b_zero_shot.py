import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def incorrect_sample_from_beta(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
    sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
    return sample_alpha / (sample_alpha + sample_beta)

def correct_sample_from_beta(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, alpha, beta=1.0)
    sample_beta = tf.random.gamma(shape, beta, beta=1.0)
    return sample_alpha / (sample_alpha + sample_beta)

shape = tf.shape(tf.range(100000))
incorrect = incorrect_sample_from_beta(0.2, 0.2, shape).numpy()
correct = correct_sample_from_beta(0.2, 0.2, shape).numpy()
reference = np.random.beta(0.2, 0.2, 100000)

plt.figure(figsize=(10, 6))
sns.kdeplot(incorrect, clip=(0, 1), label='Incorrect')
sns.kdeplot(correct, clip=(0, 1), label='Correct')
sns.kdeplot(reference, clip=(0, 1), label='Reference (numpy.random.beta)')
plt.legend()
plt.show()