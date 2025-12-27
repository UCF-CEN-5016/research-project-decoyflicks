Here's the reproduction code:

```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Minimal setup
tf.random.set_seed(0)
np.random.seed(0)

# Triggering conditions: alpha=0.2, beta=0.2, shape=(1000,)
x = tfm.vision.augment.MixupAndCutmix._sample_from_beta(0.2, 0.2, (1000,))
print(x.numpy())

# Plotting the results
sns.set()
plt.figure(figsize=(10,5))
sns.kdeplot(x.numpy(), clip=(0,1), label='tfm.vision.augment.MixupAndCutmix._sample_from_beta')
x = np.random.beta(0.2, 0.2, 100000)
sns.kdeplot(x, clip=(0,1), label='np.random.beta')
plt.legend()
plt.show()

```

Note that this code requires TensorFlow and NumPy to be installed.

