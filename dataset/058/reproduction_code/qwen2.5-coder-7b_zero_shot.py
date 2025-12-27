import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import RetinaNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def create_data_generators(train_directory: str, validation_directory: str):
    """Create image data generators for training and validation directories."""
    train_generator = ImageDataGenerator().flow_from_directory(train_directory, class_mode='categorical')
    validation_generator = ImageDataGenerator().flow_from_directory(validation_directory, class_mode='categorical')
    return train_generator, validation_generator


def build_model():
    """Instantiate the RetinaNet model."""
    return RetinaNet(weights=None)


def split_data(dataset, labels, test_size: float = 0.2, random_state: int = 42):
    """Split dataset and labels into training and validation subsets."""
    return train_test_split(dataset, labels, test_size=test_size, random_state=random_state)


def get_callbacks():
    """Prepare callbacks for training."""
    return [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]


def run_training(model, training_data, validation_data, epochs: int = 1, callbacks=None):
    """Train the model with provided datasets and callbacks."""
    model.fit(
        training_data.take(20),
        validation_data=validation_data.take(20),
        epochs=epochs,
        callbacks=callbacks
    )


def main():
    # Dataset paths and class names
    train_dir = 'path/to/train/directory'
    val_dir = 'path/to/validation/directory'
    class_names = ['class1', 'class2', 'class3']

    # Prepare data generators
    train_generator, validation_generator = create_data_generators(train_dir, val_dir)

    # Build model
    model = build_model()

    # Split data (preserving original logic)
    train_data, val_data, train_labels, val_labels = split_data(train_generator, class_names)

    # Callbacks
    callbacks = get_callbacks()

    # Train the model
    run_training(model, train_data, val_data, epochs=1, callbacks=callbacks)


if __name__ == '__main__':
    main()