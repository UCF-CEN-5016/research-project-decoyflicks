import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use the buggy _sample_from_beta function
a = tf.range(1000)
x = tfm.vision.augment.MixupAndCutmix._sample_from_beta(0.2, 0.2, tf.shape(a)).numpy()
sns.kdeplot(x, clip=(0, 1))
plt.show()

# Compare with correct implementation
x = np.random.beta(0.2, 0.2, 100000)
sns.kdeplot(x, clip=(0, 1))
plt.show()