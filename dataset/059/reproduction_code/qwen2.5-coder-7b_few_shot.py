import tensorflow as tf
from keras_cv import models
from typing import Any, Optional


def create_resnet50v2_model(
    input_shape: tuple = (224, 224, 3),
    include_top: bool = True,
    weights: Optional[str] = None,
) -> tf.keras.Model:
    """
    Construct a ResNet50V2 model with the given configuration.
    """
    return models.ResNet50V2(input_shape=input_shape, include_top=include_top, weights=weights)


def run_training(model: tf.keras.Model, train_images: Any, train_labels: Any, epochs: int = 1) -> None:
    """
    Compile and train the provided model on the given data.
    """
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(train_images, train_labels, epochs=epochs)


if __name__ == "__main__":
    model = create_resnet50v2_model()
    train_images = ...
    train_labels = ...
    run_training(model, train_images, train_labels)