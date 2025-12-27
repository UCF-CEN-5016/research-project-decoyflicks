import tensorflow as tf
from typing import Any

def build_model() -> tf.keras.Model:
    """
    Construct and return a simple Keras Sequential model with a Conv2D output.
    Input shape: (2, 2, 1)
    Output: 3 channels with softmax activation
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2, 2, 1)),
        tf.keras.layers.Conv2D(3, (2, 2), activation='softmax')
    ])

def compile_and_train(model: tf.keras.Model, features: Any, targets: Any, loss: str) -> None:
    """
    Compile the model with Adam optimizer and the specified loss, then fit on the provided data.
    """
    model.compile(optimizer='adam', loss=loss)
    model.fit(x=features, y=targets)

# Instantiate the model
model = build_model()

# Labels (kept as in original code)
labels = tf.constant([[0, 1, 2]])

# Train with categorical_crossentropy (as in original)
compile_and_train(model, tf.random.normal([1, 2, 2, 1]), labels, 'categorical_crossentropy')

# Train with sparse_categorical_crossentropy (as in original)
compile_and_train(model, tf.random.normal([1, 2, 2, 1]), labels, 'sparse_categorical_crossentropy')