import tensorflow_models as tfm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

alpha = 0.2
beta = 0.2
shape = (1000,)
mixup_samples = tfm.vision.augment.MixupAndCutmix._sample_from_beta(alpha, beta, shape).numpy()

reference_samples = np.random.beta(alpha, beta, 100000)

plt.figure()
sns.kdeplot(mixup_samples, label='Mixup Samples', clip=(0, 1))
sns.kdeplot(reference_samples, label='Reference Samples', clip=(0, 1))
plt.xlim(0, 1)
plt.title('Mixup Samples vs Reference Samples')
plt.legend()
plt.show()

assert not np.isclose(np.mean(mixup_samples), 0.5, atol=0.1)