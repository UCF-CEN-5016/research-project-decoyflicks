import tensorflow as tf
from keras_cv import YOLOv8
from keras_cv.utils.coco_metrics import BoxCOCOMetricsCallback

# Ensure TF is correctly configured for hardware support (e.g., GPU)
tf.config.set_visible_devices([], "GPU")


def read_image_and_label(image_path: tf.Tensor):
    """Read an image file and return it with a dummy label."""
    image = tf.io.read_file(image_path)
    return image, tf.constant([0])


def create_dataset(image_paths, batch_size: int):
    """Create a batched tf.data.Dataset from a list of image paths."""
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(read_image_and_label)
    ds = ds.batch(batch_size)
    return ds


def main():
    # Minimal dataset setup
    training_image_paths = [
        "path/to/training/image1.jpg",
        "path/to/training/image2.jpg",
    ]
    validation_image_paths = ["path/to/val/image3.jpg"]

    train_dataset = create_dataset(training_image_paths, batch_size=2)
    val_dataset = create_dataset(validation_image_paths, batch_size=1)

    # Example YOLOv8 model (replace with actual trained model)
    yolo_model = YOLOv8("yolov8n")
    yolo_model.train(
        train_dataset,
        epochs=1,
        callbacks=[BoxCOCOMetricsCallback(val_dataset, "model.h5")],
    )

    # Verify metrics computation
    coco_metrics_cb = BoxCOCOMetricsCallback(val_dataset, "model.h5")
    coco_metrics_cb.on_epoch_end(0, None)


if __name__ == "__main__":
    main()