import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def bad_sample_from_beta(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
    sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
    return sample_alpha / (sample_alpha + sample_beta)

shape = (100000,)
x_bad = bad_sample_from_beta(0.2, 0.2, shape).numpy()
x_true = np.random.beta(0.2, 0.2, 100000)

sns.kdeplot(x_bad, clip=(0,1), label='Bad tf gamma beta')
sns.kdeplot(x_true, clip=(0,1), label='Numpy beta')
plt.legend()
plt.show()