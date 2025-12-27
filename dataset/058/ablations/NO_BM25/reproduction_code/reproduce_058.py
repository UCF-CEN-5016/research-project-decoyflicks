import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Placeholder functions for undefined variables
def swap_xy(bbox):
    # Swap x and y coordinates of bounding boxes
    bbox[:, [0, 1]] = bbox[:, [1, 0]]
    return bbox

def random_flip_horizontal(image, bbox):
    # Randomly flip the image and bounding boxes horizontally
    if np.random.rand() > 0.5:
        image = tf.image.flip_left_right(image)
        bbox[:, [0, 2]] = 1 - bbox[:, [2, 0]]  # Update x-coordinates
    return image, bbox

def resize_and_pad_image(image):
    # Resize and pad the image to a fixed size
    target_size = (800, 1333)
    image = tf.image.resize(image, target_size)
    image_shape = tf.shape(image)
    return image, image_shape, None

def convert_to_xywh(bbox):
    # Convert bounding boxes from (x1, y1, x2, y2) to (x, y, width, height)
    x = bbox[:, 0]
    y = bbox[:, 1]
    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]
    return tf.stack([x, y, width, height], axis=-1)

class LabelEncoder:
    def encode_batch(self, batch):
        # Dummy implementation for label encoding
        return batch

def get_backbone():
    # Placeholder for backbone model
    return keras.applications.ResNet50(input_shape=(800, 1333, 3), include_top=False)

class RetinaNetLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        # Dummy implementation for loss calculation
        return tf.reduce_mean(y_pred)

class RetinaNet(tf.keras.Model):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    def call(self, inputs):
        # Dummy implementation for model call
        return self.backbone(inputs)

def generate_synthetic_data(num_images=100, num_boxes=5):
    images = []
    bboxes = []
    class_ids = []
    for _ in range(num_images):
        image = np.random.rand(800, 1333, 3).astype(np.float32)
        boxes = np.random.rand(num_boxes, 4)  # Random boxes
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]  # Ensure boxes are valid
        boxes = np.clip(boxes, 0, 1)  # Clip to [0, 1]
        class_id = np.random.randint(0, 80, size=num_boxes)
        images.append(image)
        bboxes.append(boxes)
        class_ids.append(class_id)
    return images, bboxes, class_ids

images, bboxes, class_ids = generate_synthetic_data()

def preprocess_data(sample):
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)
    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)
    bbox = tf.stack([
        bbox[:, 0] * image_shape[1],
        bbox[:, 1] * image_shape[0],
        bbox[:, 2] * image_shape[1],
        bbox[:, 3] * image_shape[0],
    ], axis=-1)
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id

batch_size = 2
num_classes = 80

train_dataset = tf.data.Dataset.from_tensor_slices({
    "image": images,
    "objects": {
        "bbox": bboxes,
        "label": class_ids
    }
})
train_dataset = train_dataset.map(preprocess_data)
train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)

label_encoder = LabelEncoder()
train_dataset = train_dataset.map(label_encoder.encode_batch)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

# The following line is kept to reproduce the bug
model.fit(
    train_dataset.take(1),
    epochs=1,
    verbose=1,
)