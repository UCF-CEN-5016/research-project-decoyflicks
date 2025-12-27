import tensorflow_models as tfm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

alpha = 0.2
beta = 0.2

tensor = tf.range(1000)

samples = tfm.vision.augment.MixupAndCutmix._sample_from_beta(alpha, beta, tensor).numpy()

sns.kdeplot(samples[samples >= 0] and samples[samples <= 1], clip=(0, 1))
plt.show()

reference_samples = np.random.beta(alpha, beta, size=1000)

sns.kdeplot(reference_samples[reference_samples >= 0] and reference_samples[reference_samples <= 1], clip=(0, 1))
plt.show()