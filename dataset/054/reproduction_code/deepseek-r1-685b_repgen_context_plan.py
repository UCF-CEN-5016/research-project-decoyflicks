import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np

# Assuming 'subclassing_conv_layers.py' exists in the same directory or is importable
# This custom layer is critical to the original code's intent.
try:
    from subclassing_conv_layers import StandardizedConv2DWithOverride
except ImportError:
    print("Error: 'subclassing_conv_layers.py' not found or StandardizedConv2DWithOverride not defined within it.")
    print("Please ensure this file is in the same directory or accessible in your Python path.")
    print("For demonstration purposes, a dummy StandardizedConv2DWithOverride will be used.")
    class StandardizedConv2DWithOverride(layers.Conv2D):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print("Using dummy StandardizedConv2DWithOverride.")


class TrainingConfig:
    def __init__(self):
        self.batch_size = 128
        self.img_rows = 28
        self.img_cols = 28
        self.num_classes = 10
        self.epochs = 20
        self.train_subset_size = 50000

def load_and_preprocess_mnist_data(config: TrainingConfig):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape((x_train.shape[0], config.img_rows, config.img_cols, 1))
    x_test = x_test.reshape((x_test.shape[0], config.img_rows, config.img_cols, 1))

    y_train = to_categorical(y_train, num_classes=config.num_classes)
    y_test = to_categorical(y_test, num_classes=config.num_classes)

    x_train_subset = x_train[:config.train_subset_size]
    y_train_subset = y_train[:config.train_subset_size]
    
    return (x_train_subset, y_train_subset), (x_test, y_test)

def build_cnn_model(config: TrainingConfig):
    model = models.Sequential([
        StandardizedConv2DWithOverride(32, (3, 3), activation='relu', input_shape=(config.img_rows, config.img_cols, 1)),
        layers.MaxPooling2D((2, 2)),
        StandardizedConv2DWithOverride(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(config.num_classes, activation='softmax')
    ])
    return model

def compile_and_train_model(model: tf.keras.Model, train_data: tuple, test_data: tuple, config: TrainingConfig):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    x_train_subset, y_train_subset = train_data
    x_test, y_test = test_data

    model.fit(
        x_train_subset,
        y_train_subset,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_test, y_test)
    )

def main():
    config = TrainingConfig()

    (x_train_subset, y_train_subset), (x_test, y_test) = load_and_preprocess_mnist_data(config)
    model = build_cnn_model(config)
    compile_and_train_model(model, (x_train_subset, y_train_subset), (x_test, y_test), config)

if __name__ == "__main__":
    main()
