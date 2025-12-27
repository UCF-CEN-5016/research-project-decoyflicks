# Install dependencies
# pip install tensorflow==2.13.10
# pip install tf-models-nightly
# pip install matplotlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from official.nlp.modeling.layers.tn_expand_condense import TNExpandCondense

print('Current backend:', plt.get_backend())

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave')
plt.show()

import official.nlp.modeling.layers.tn_expand_condense as tn_layers

print('Current backend after import:', plt.get_backend())

plt.plot(x, y)
plt.title('Sine Wave After Import')
plt.show()

assert plt.get_backend() == 'agg'