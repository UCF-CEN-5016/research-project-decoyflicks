import tensorflow as tf
import keras_cv

# Create a dummy dataset with invalid class labels
def create_invalid_dataset():
    # Create a dataset with one image and one box with class ID 80 (assuming num_classes=80)
    images = tf.data.Dataset.from_tensor_slices([tf.zeros([100, 100, 3])])
    boxes = tf.data.Dataset.from_tensor_slices([tf.constant([[0., 0., 100., 100.]], dtype=tf.float32)])
    classes = tf.data.Dataset.from_tensor_slices([tf.constant([80], dtype=tf.int32)])
    dataset = tf.data.Dataset.zip((images, boxes, classes))
    return dataset

# Create the invalid dataset
invalid_dataset = create_invalid_dataset()

# Define the RetinaNet model with num_classes=80
model = keras_cv.models.RetinaNet(
    num_classes=80,
    backbone='resnet50',
    weights='coco'
)

# Compile the model
model.compile(optimizer='adam', loss=model.compute_loss())

# Train the model (this will trigger the error)
try:
    model.fit(invalid_dataset, epochs=1)
except Exception as e:
    print("Error occurred:", e)