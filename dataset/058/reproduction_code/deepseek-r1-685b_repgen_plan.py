import tensorflow as tf
from keras_cv.models import RetinaNet
import numpy as np

def create_dataset():
    # Create a minimal dataset with invalid bounding boxes
    images = tf.random.uniform((2, 512, 512, 3))  # 2 sample images
    # Problematic boxes: class_id=0 (background) with invalid coordinates
    boxes = tf.constant([
        [[0, 0, 0, 0, 0]],  # Background box with zero coordinates
        [[1, -10, -10, 600, 600]]  # Box outside image bounds
    ], dtype=tf.float32)

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((images, boxes))
    ds = ds.batch(2)
    return ds

def train_model(ds):
    # Create RetinaNet model
    model = RetinaNet(
        classes=1,  # Just one class (plus background)
        bounding_box_format="xyxy"
    )

    # Compile the model
    model.compile(
        optimizer="adam",
        classification_loss="focal",
        box_loss="smoothl1"
    )

    # Attempt training
    model.fit(ds, epochs=1)

if __name__ == "__main__":
    dataset = create_dataset()
    train_model(dataset)