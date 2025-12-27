import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow_models.vision.ops import MixupAndCutmix

a = 0.2
b = 0.2
x = MixupAndCutmix._sample_from_beta(a, b, (100000,))
sns.kdeplot(x, clip=(0, 1))
y = np.random.beta(a, b, 100000)
sns.kdeplot(y, clip=(0, 1))
plt.show()