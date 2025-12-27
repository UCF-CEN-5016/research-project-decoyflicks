import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

NUM_EPOCHS = 100
LOG_DIR = 'logs'
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 10
DROPOUT_RATE = 0.2


def build_cnn_model() -> tf.keras.Model:
    """Constructs and returns a simple CNN model."""
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(NUM_CLASSES)
    ])


def get_training_callbacks():
    """Creates and returns the list of callbacks for training."""
    return [TensorBoard(log_dir=LOG_DIR), EarlyStopping()]


def run_training() -> tf.keras.Model:
    """Builds the model, fits it, and returns the trained model."""
    model = build_cnn_model()
    callbacks = get_training_callbacks()
    model.fit(x=None, epochs=NUM_EPOCHS, validation_data=None, validation_freq=1, callbacks=callbacks)
    return model


model = run_training()