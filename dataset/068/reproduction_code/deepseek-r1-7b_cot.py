import tensorflow as tf
from keras_cv import YOLOv8
from keras_cv.utils.coco_metrics import BoxCOCOMetricsCallback

# Ensure TF is correctly configured for hardware support (e.g., GPU)
tf.config.set_visible_devices([], 'GPU')

# Minimal dataset setup
def load_image_label(image_path):
    image = tf.io.read_file(image_path)
    return image, tf.constant([0])

train_paths = ["path/to/training/image1.jpg", "path/to/training/image2.jpg"]
val_paths = ["path/to/val/image3.jpg"]

train_ds = tf.data.Dataset.from_tensor_slices(train_paths).map(load_image_label).batch(2)
val_ds = tf.data.Dataset.from_tensor_slices(val_paths).map(load_image_label).batch(1)

# Example YOLOv8 model (replace with actual trained model)
model = YOLOv8('yolov8n')
model.train(
    train_ds,
    epochs=1,
    callbacks=[BoxCOCOMetricsCallback(val_ds, 'model.h5')],
)

# Verify metrics computation
metrics_callback = BoxCOCOMetricsCallback(val_ds, 'model.h5')
metrics_callback.on_epoch_end(0, None)