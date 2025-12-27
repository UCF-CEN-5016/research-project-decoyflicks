import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sample_from_beta(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
    sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
    return sample_alpha / (sample_alpha + sample_beta)

shape = tf.shape(tf.range(1000))
samples_incorrect = sample_from_beta(0.2, 0.2, shape).numpy()
samples_correct = np.random.beta(0.2, 0.2, 100000)

sns.kdeplot(samples_incorrect, clip=(0, 1), label='Incorrect')
sns.kdeplot(samples_correct, clip=(0, 1), label='Correct')
plt.legend()
plt.show()