import tensorflow as tf
import keras
import keras_cv

# Create mock dataset with invalid label formatting
def create_mock_dataset():
    # Mock images (batch_size=2, height=300, width=300, channels=3)
    images = tf.random.uniform(shape=(2, 300, 300, 3))
    
    # Invalid labels: class IDs exceed expected vocabulary size (num_classes=2)
    # This would cause indices to be out of bounds during label encoding
    class_ids = tf.constant([[3, 1, 0], [0, 2, 1]])  # Contains invalid class ID 3
    bounding_boxes = tf.random.uniform(shape=(2, 3, 4))  # Valid box coordinates
    
    # Create dataset with invalid labels
    dataset = tf.data.Dataset.from_tensor_slices((images, class_ids, bounding_boxes))
    return dataset

# Create model with expected class vocabulary size (num_classes=2)
retina_net = keras_cv.models.RetinaNet(
    num_classes=2,
    backbone=keras_cv.models.RetinaNetBackbone('resnet50'),
    pretrained_backbone=False
)

# Create dataset with invalid labels
train_ds = create_mock_dataset()

# Attempt to train with invalid labels
retina_net.fit(
    train_ds,
    epochs=1
)