import tensorflow_models as tfm  
import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  

alpha = 0.2
beta = 0.2
shape = (1000,)
sample = tfm.vision.augment.MixupAndCutmix._sample_from_beta(alpha, beta, shape).numpy()

reference_sample = np.random.beta(alpha, beta, 100000)
sns.kdeplot(sample, clip=(0, 1))
sns.kdeplot(reference_sample, clip=(0, 1))
plt.show()

assert np.all(sample >= 0) and np.all(sample <= 1), "Sample contains values outside the expected range of [0, 1]"