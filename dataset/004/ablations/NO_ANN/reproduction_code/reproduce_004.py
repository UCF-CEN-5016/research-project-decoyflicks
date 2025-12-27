import tensorflow as tf
import tensorflow_models as tfm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

alpha = 0.2
beta = 0.2
shape = tf.shape(tf.range(1000))

x = tfm.vision.augment.MixupAndCutmix._sample_from_beta(alpha, beta, shape)
x_numpy = x.numpy()

reference_sample = np.random.beta(alpha, beta, 100000)

sns.kdeplot(x_numpy, clip=(0, 1))
sns.kdeplot(reference_sample, clip=(0, 1))
plt.show()