import tensorflow as tf
from official.vision import keras_cv
from official.vision.dataloaders import segmentation_input

# Create the SegmentationModel
model = keras_cv.models.SegmentationModel(
    backbone='resnet50',
    num_classes=2,  # Binary segmentation
    backbone_weights=None
)

# Define a function to create the dummy dataset
def create_dummy_dataset(num_samples=10):
    images = tf.random.uniform((num_samples, 256, 256, 3))
    masks = tf.random.uniform((num_samples, 256, 256), maxval=2, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((images, masks)).repeat()

# Create dummy training and validation datasets
train_ds = create_dummy_dataset()
val_ds = create_dummy_dataset()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=5,
    validation_steps=2
)

# Display training history
print("\nTraining history:")
print(history.history)