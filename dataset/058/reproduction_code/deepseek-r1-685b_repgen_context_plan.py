import tensorflow as tf
from keras_cv import models

def create_dataset(num_samples):
    images = tf.random.uniform((num_samples, 512, 512, 3))
    boxes = tf.constant([
        [[0, 0, 0, 0, 0]],  # Background box with zero coordinates
        [[1, -10, -10, 600, 600]]  # Box outside image bounds
    ], dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((images, boxes))
    return ds.batch(num_samples)

def train_retinanet(dataset):
    model = models.RetinaNet(classes=1, bounding_box_format="xyxy")
    model.compile(optimizer="adam", classification_loss="focal", box_loss="smoothl1")
    model.fit(dataset, epochs=1)

if __name__ == "__main__":
    dataset = create_dataset(2)
    train_retinanet(dataset)