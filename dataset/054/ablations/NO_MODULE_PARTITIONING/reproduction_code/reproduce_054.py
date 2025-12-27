import numpy as np
import tensorflow as tf
from keras import layers, Model, Sequential

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 32
EPOCHS = 20
INPUT_SHAPE = (28, 28, 28, 1)
NUM_CLASSES = 11

train_data = np.random.rand(1000, 28, 28, 28, 1)
train_labels = np.random.randint(0, NUM_CLASSES, 1000)
valid_data = np.random.rand(200, 28, 28, 28, 1)
valid_labels = np.random.randint(0, NUM_CLASSES, 200)
test_data = np.random.rand(200, 28, 28, 28, 1)
test_labels = np.random.randint(0, NUM_CLASSES, 200)

model = Sequential()
model.add(layers.Input(shape=INPUT_SHAPE))
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu'))
model.add(layers.GlobalAvgPool3D())
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=EPOCHS, validation_data=(test_data, test_labels))