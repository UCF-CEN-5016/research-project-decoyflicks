import numpy as np
from keras_cv.models.object_detection import RetinaNet

# Define the Retina Net model
model = RetinaNet()

# Define sample data
train_data = {
    "image": np.random.rand(32, 224, 224, 3),
    "bboxes": np.random.rand(32, 10, 4),  # Bounding boxes with (x1, y1, x2, y2) format
    "class_ids": np.random.randint(0, 20, size=(32, 10)),  # Class IDs for each bounding box
}

# Train the model
model.fit(train_data["image"], train_data["bboxes"], epochs=1)