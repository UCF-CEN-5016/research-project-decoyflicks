import tensorflow as tf
from keras import layers

# Define a version of keras that causes the bug, e.g., 2.13.0
# Set up the environment by running 'pip install keras==2.13.0'

batch_size = 64

# Prepare random input data with shape (batch_size, height=28, width=28, channels=1)
x_train = tf.random.normal((batch_size, 28, 28, 1))

# Load and preprocess the MNIST dataset using keras.datasets.mnist.load_data() and normalize the data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0

# Create a simple Sequential model with layers: InputLayer, StandardizedConv2DWithOverride, MaxPooling2D, StandardizedConv2DWithCall, Flatten, Dropout, Dense
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model with loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Define labels for y_train using keras.utils.to_categorical() with num_classes=10
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Train the model on x_train and y_train for 5 epochs using batch_size of 64
model.fit(x_train, y_train, epochs=5, batch_size=batch_size)