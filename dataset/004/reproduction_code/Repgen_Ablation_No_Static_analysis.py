import tensorflow as tf
from tfm.vision.augment import _sample_from_beta

alpha = 0.2
beta = 0.2
shape = (1000,)

sampled_values = _sample_from_beta(alpha, beta, shape)

# Verifying the distribution of sampled values
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(sampled_values)
plt.show()

# Comparing with numpy's implementation for reference
import numpy as np
expected_samples = np.random.beta(alpha, beta, size=shape)

# Asserting the difference
assert not np.array_equal(sampled_values, expected_samples), "The sampled values match the expected shape of a beta distribution"

# Printing out a subset of the sampled values
print(sampled_values[:10])