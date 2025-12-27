import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def _sample_from_beta(alpha, beta, shape):
    sample_alpha = tf.random.gamma(shape, alpha, beta=1.0)
    sample_beta = tf.random.gamma(shape, beta, beta=1.0)
    return sample_alpha / (sample_alpha + sample_beta)

alpha = 0.2
beta = 0.2
tensor_shape = (1000,)

sampled_values = _sample_from_beta(alpha, beta, tensor_shape)
sampled_values_np = sampled_values.numpy()

sns.kdeplot(sampled_values_np, clip=(0, 1))
plt.show()

expected_samples = np.random.beta(alpha, beta, size=100000)
sns.kdeplot(expected_samples, clip=(0, 1))
plt.show()