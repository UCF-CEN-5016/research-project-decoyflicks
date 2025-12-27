import numpy as np
from tensorflow import keras
from keras.utils import to_categorical

batch_size = 128
input_shape = (28, 28, 1)

x_train = np.random.uniform(size=(batch_size, *input_shape)).astype("float32") / 255
y_train = np.random.randint(0, 10, size=batch_size)
y_train = to_categorical(y_train, num_classes=10)

try:
    from keras.utils import FeatureSpace
except ImportError as e:
    print(e)