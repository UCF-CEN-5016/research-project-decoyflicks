import tensorflow_models as tfm  # Official vision augment module
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

alpha = 0.2
beta = 0.2
shape = tf.shape(tf.range(1000))

mixup_samples = tfm.vision.augment.MixupAndCutmix._sample_from_beta(alpha, beta, shape).numpy()
reference_samples = np.random.beta(alpha, beta, 100000)

plt.figure()
sns.kdeplot(mixup_samples, clip=(0, 1))
sns.kdeplot(reference_samples, clip=(0, 1))
plt.show()