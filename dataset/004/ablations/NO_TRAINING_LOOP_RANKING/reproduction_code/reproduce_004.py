import tensorflow_models as tfm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

alpha = 0.2
beta = 0.2
shape = (1000,)

original_sample = tfm.vision.augment.MixupAndCutmix._sample_from_beta(alpha, beta, shape).numpy()
reference_sample = np.random.beta(alpha, beta, 100000)

sns.kdeplot(original_sample, label='Original Sample', clip=(0, 1))
sns.kdeplot(reference_sample, label='Reference Sample', clip=(0, 1))
plt.title('Comparison of Original and Reference Beta Samples')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

assert not np.allclose(np.histogram(original_sample, bins=30, density=True)[0], 
                       np.histogram(reference_sample, bins=30, density=True)[0]), "Distributions are similar, bug not present."