import os
import keras
from keras import layers
from keras.layers import TextVectorization

os.environ['KERAS_BACKEND'] = 'tensorflow'

def generate_random_text_data(num_samples=1000, sequence_length=20):
    import numpy as np
    return np.random.randint(0, 256, size=(num_samples, sequence_length))

x_train = generate_random_text_data()
batch_size = 32

model = keras.Sequential([
    layers.Input(shape=(None, 20)),
    layers.Dense(64, activation='relu')
])

try:
    from keras import ops
except ImportError as e:
    print(f"ImportError: {e}")

model.summary()