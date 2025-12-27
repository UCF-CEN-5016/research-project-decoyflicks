import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MixupAndCutmix:
    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

a = tf.range(1000)
x = MixupAndCutmix._sample_from_beta(0.2, 0.2, tf.shape(tf.range(1000))).numpy()
sns.kdeplot(x, clip=(0, 1))
x = np.random.beta(0.2, 0.2, 100000)
sns.kdeplot(x, clip=(0, 1))
plt.show()