import tensorflow as tf
from efficientdet import EfficientDet

DEFAULT_BACKBONE = 'd1'
DEFAULT_NUM_CLASSES = 81
DEFAULT_EPOCHS = 5
DEFAULT_NUM_SAMPLES = 1
DEFAULT_IMAGE_SHAPE = (224, 224, 3)
DEFAULT_LABEL_SHAPE = (2,)


def build_dataset(num_samples: int = DEFAULT_NUM_SAMPLES,
                  image_shape: tuple = DEFAULT_IMAGE_SHAPE,
                  label_shape: tuple = DEFAULT_LABEL_SHAPE) -> tf.data.Dataset:
    images = tf.zeros((num_samples,) + image_shape)
    labels = tf.zeros((num_samples,) + label_shape)
    return tf.data.Dataset.from_tensor_slices((images, labels))


def build_model(backbone_name: str = DEFAULT_BACKBONE,
                num_classes: int = DEFAULT_NUM_CLASSES) -> EfficientDet:
    return EfficientDet(backbone_name=backbone_name, num_classes=num_classes)


def compile_model(model: EfficientDet,
                  optimizer: str = 'adam',
                  loss: str = 'categorical_crossentropy') -> None:
    model.compile(optimizer=optimizer, loss=loss)


def train_model(model: EfficientDet,
                train_ds: tf.data.Dataset,
                val_ds: tf.data.Dataset,
                epochs: int = DEFAULT_EPOCHS):
    return model.fit(train_ds, epochs=epochs, validation_data=val_ds)


def main():
    model = build_model()
    train_dataset = build_dataset()
    validation_dataset = build_dataset()
    compile_model(model)
    history = train_model(model, train_dataset, validation_dataset)
    return history


if __name__ == '__main__':
    history = main()