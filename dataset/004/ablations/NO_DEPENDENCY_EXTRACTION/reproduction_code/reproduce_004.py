import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from official.vision.dataloaders import tf_example_decoder

class MixupAndCutmix:
    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        return np.random.beta(alpha, beta, shape)

alpha = 0.2
beta = 0.2
shape = (1000,)
sample = MixupAndCutmix._sample_from_beta(alpha, beta, shape)

reference_sample = np.random.beta(alpha, beta, 100000)

sns.kdeplot(sample, clip=(0, 1))
sns.kdeplot(reference_sample, clip=(0, 1))
plt.show()

assert np.max(sample) > 1 or np.min(sample) < 0