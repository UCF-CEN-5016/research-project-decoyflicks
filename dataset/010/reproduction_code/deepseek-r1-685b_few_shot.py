import tensorflow as tf
from official.vision import keras_cv
from official.vision.dataloaders import segmentation_input

# Minimal model setup
model = keras_cv.models.SegmentationModel(
    backbone='resnet50',
    num_classes=2,  # Binary segmentation
    backbone_weights=None
)

# Dummy dataset
def dummy_dataset():
    return tf.data.Dataset.from_tensors((
        tf.random.uniform((1, 256, 256, 3)),  # Image
        tf.random.uniform((1, 256, 256), maxval=2, dtype=tf.int32)  # Mask
    )).repeat(10)

train_ds = dummy_dataset()
val_ds = dummy_dataset()

# Training setup
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Training with validation - will show zero validation loss
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=5,
    validation_steps=2
)

# Shows validation_loss: 0.0 despite training
print("\nTraining history:")
print(history.history)