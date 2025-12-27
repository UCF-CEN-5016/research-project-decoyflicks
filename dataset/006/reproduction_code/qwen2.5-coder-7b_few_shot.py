import tensorflow as tf
from typing import Tuple

def build_regression_model(input_dim: int = 32,
                           hidden_units: int = 64,
                           output_units: int = 10) -> tf.keras.Model:
    """
    Construct a simple feedforward regression model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_units)
    ])
    return model

def compile_model(model: tf.keras.Model,
                  optimizer: str = 'adam',
                  loss: str = 'mse') -> None:
    """
    Compile the provided Keras model with given optimizer and loss.
    """
    model.compile(optimizer=optimizer, loss=loss)

def generate_dummy_data(num_samples: int = 128,
                        input_dim: int = 32,
                        output_dim: int = 10) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate dummy input and target tensors for training.
    """
    inputs = tf.random.normal([num_samples, input_dim])
    targets = tf.random.normal([num_samples, output_dim])
    return inputs, targets

def run_training(epochs: int = 1) -> None:
    """
    Build, compile, generate data, and train the model.
    """
    model = build_regression_model()
    compile_model(model)
    inputs, targets = generate_dummy_data()
    model.fit(inputs, targets, epochs=epochs)

if __name__ == '__main__':
    run_training()