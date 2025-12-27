import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten


def build_conv_model(img_height: int, img_width: int, channels: int) -> Model:
    """Builds a small convolutional model that outputs a flattened feature vector."""
    inputs = Input(shape=(img_height, img_width, channels))
    x = Conv2D(8, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    return Model(inputs=inputs, outputs=x)


def generate_random_dataset(samples: int, img_height: int, img_width: int, channels: int, num_classes: int):
    """Generates a random dataset of images and integer labels."""
    X = np.random.rand(samples, img_height, img_width, channels)
    y = np.random.randint(0, num_classes, (samples,))
    return X, y


def train_model(model: Model, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
    """Compiles and trains the model on provided data."""
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X, y, epochs=epochs)


def main() -> None:
    img_rows, img_cols, channels = 32, 32, 3
    num_classes = 10
    num_samples = 1000

    model = build_conv_model(img_rows, img_cols, channels)
    X_train, y_train = generate_random_dataset(num_samples, img_rows, img_cols, channels, num_classes)
    train_model(model, X_train, y_train, epochs=1)


if __name__ == '__main__':
    main()