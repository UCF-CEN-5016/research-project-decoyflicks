import tensorflow_models as tfm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

alpha = 0.2
beta = 0.2
shape = (1000,)

sampled_values = tfm.vision.augment.MixupAndCutmix._sample_from_beta(alpha, beta, shape).numpy()
reference_sample = np.random.beta(alpha, beta, 100000)

plt.figure()
sns.kdeplot(sampled_values, clip=(0, 1))
sns.kdeplot(reference_sample, clip=(0, 1))
plt.title('Comparison of MixUp and CutMix Samples')
plt.xlabel('Sample Value')
plt.ylabel('Density')
plt.show()