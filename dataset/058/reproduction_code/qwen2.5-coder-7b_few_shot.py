import tensorflow as tf
from tensorflow.keras import layers
from keras_cv import models


def prepare_mock_dataset() -> tf.data.Dataset:
    """Create a small mock dataset for training."""
    images = tf.random.uniform(shape=(2, 300, 300, 3))
    class_ids = tf.constant([[3, 1, 0], [0, 2, 1]])
    bounding_boxes = tf.random.uniform(shape=(2, 3, 4))
    dataset = tf.data.Dataset.from_tensor_slices((images, class_ids, bounding_boxes))
    return dataset


def build_retinanet_model(num_classes: int) -> tf.keras.Model:
    """Construct a RetinaNet model with the specified number of classes."""
    backbone = models.RetinaNetBackbone('USE_BACKBONE_MODEL')
    model = models.RetinaNet(
        num_classes=num_classes,
        backbone=backbone,
        pretrained_backbone=False
    )
    return model


def get_loss_fn(num_classes: int):
    """Return the RetinaNet loss for the given number of classes."""
    return models.retinanet.RetinaNetLoss(num_classes=num_classes)


def get_optimizer():
    """Create and return the optimizer."""
    return tf.keras.optimizers.Adam()


def main():
    dataset = prepare_mock_dataset()
    model = build_retinanet_model(num_classes=2)
    loss_fn = get_loss_fn(num_classes=2)
    optimizer = get_optimizer()

    model.compile(loss=loss_fn, optimizer=optimizer)

    model.fit(dataset, epochs=1)


if __name__ == "__main__":
    main()