import tensorflow as tf
from tensorflow import keras
from keras_cv.models import YoloV8
from keras_cv.metrics import COCOMetrics
from tensorflow.keras.datasets import mnist
import numpy as np

# Set up minimal environment
np.random.seed(42)
tf.random.set_seed(42)

# Create a dummy dataset (replace with your actual dataset)
def create_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Create dummy bounding boxes and class labels
    train_boxes = np.random.rand(len(x_train), 4, 4)
    test_boxes = np.random.rand(len(x_test), 4, 4)
    train_labels = np.random.randint(0, 10, size=len(x_train))
    test_labels = np.random.randint(0, 10, size=len(x_test))
    
    return (x_train, train_boxes, train_labels), (x_test, test_boxes, test_labels)

# Load the dataset
(train_images, train_boxes, train_labels), (test_images, test_boxes, test_labels) = create_dataset()

# Create data generators
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_boxes, train_labels)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_boxes, test_labels)).batch(32)

# Define the YOLOV8 model
yolo = YoloV8(
    classes=10,
    bounding_box_format='xyxy',
    num_layers=2,
    box_target_generator=keras_cv.models.YoloV8.box_target_generator,
)

# Compile the model
yolo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras_cv.models.YoloV8.loss,
)

# Define the COCO metrics callback
coco_metrics_callback = COCOMetrics(
    val_ds=test_ds,
    bounding_box_format='xyxy',
    class_ids=range(10),
)

# Train the model (triggering conditions)
yolo.fit(
    train_ds,
    validation_data=test_ds,
    epochs=3,
    callbacks=[coco_metrics_callback],
)