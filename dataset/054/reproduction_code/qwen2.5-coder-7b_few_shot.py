import numpy as np
import tensorflow as tf
from typing import Tuple

NUM_TRAIN_SAMPLES = 100
NUM_TEST_SAMPLES = 20
NUM_FEATURES = 10
EPOCHS = 5


def make_random_dataset(num_samples: int, num_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random features and labels for regression."""
    features = np.random.rand(num_samples, num_features)
    labels = np.random.rand(num_samples, 1)
    return features, labels


def build_regression_model(input_dim: int) -> tf.keras.Model:
    """Create a simple feedforward regression model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(1))
    return model


def compile_and_train(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
) -> tf.keras.callbacks.History:
    """Compile the model and train with provided validation data."""
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
    return history


def main() -> None:
    x_train, y_train = make_random_dataset(NUM_TRAIN_SAMPLES, NUM_FEATURES)
    x_test, y_test = make_random_dataset(NUM_TEST_SAMPLES, NUM_FEATURES)

    model = build_regression_model(NUM_FEATURES)
    compile_and_train(model, x_train, y_train, x_test, y_test, EPOCHS)


if __name__ == "__main__":
    main()