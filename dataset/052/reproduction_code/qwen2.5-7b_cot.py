import tensorflow as tf
import numpy as np

def create_model(input_shape, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Embedding(input_dim=1000, output_dim=64),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    return model

def train_model(model, x, y, optimizer='adam', loss='sparse_categorical_crossentropy', epochs=1):
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(x, y, epochs=epochs)

# Define and train model with sequence data
sequence_model = create_model((None,), 10)
x_seq = np.random.normal(size=(32, 10))
y_seq = np.random.randint(0, 10, size=(32,))
train_model(sequence_model, x_seq, y_seq)

# Define model with image data
image_model = create_model((28, 28, 3), 10)
image_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
image_model.add(tf.keras.layers.GlobalAveragePooling2D())

# Define another model with sequence data
sequence_model_2 = create_model((None,), 10)