import numpy as np
import tensorflow as tf
from keras import layers
from keras_tuner.tuners import RandomSearch
from keras_cv.layers import ImageDataAugmentation

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create synthetic data
x_train = np.random.rand(1000, 28, 28, 1)
y_train = np.random.randint(0, 9, (1000, 1))
x_val = np.random.rand(1000, 28, 28, 1)
y_val = np.random.randint(0, 9, (1000, 1))

# Set hyperparameters
batch_size = 64
learning_rate = 0.001

# Initialize MyHyperModel class
class MyHyperModel:
    def build_model(self, hp):
        units_1 = hp.Int("units_1", 10, 40, step=10)
        units_2 = hp.Int("units_2", 10, 30, step=10)
        model = tf.keras.Sequential([
            layers.Dense(units=units_1, input_shape=(28*28,), activation='relu'),
            layers.Flatten(),
            layers.Dense(units=units_2, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, hp):
        model = self.build_model(hp)
        history = model.fit(x_train.reshape(-1, 28*28), y_train,
                            batch_size=batch_size,
                            epochs=10,
                            validation_data=(x_val.reshape(-1, 28*28), y_val))
        return history

# Initialize tuner
tuner = RandomSearch(
    MyHyperModel(),
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1
)

# Start search
try:
    tuner.search(x_train.reshape(-1, 28*28), y_train,
                 epochs=10,
                 validation_data=(x_val.reshape(-1, 28*28), y_val))
except Exception as e:
    print(e)