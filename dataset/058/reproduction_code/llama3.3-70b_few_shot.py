import tensorflow as tf
from tensorflow import keras
from keras_cv.models import RetinaNet
from keras_cv.models.object_detection import RetinaNet
from keras_cv import layers

# Define a simple RetinaNet model
model = RetinaNet(
    classes=1,
    backbone="resnet50",
    image_size=(512, 512),
    include_rescaling=True,
)

# Create a simple dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    (
        tf.random.normal([10, 512, 512, 3]),
        tf.random.uniform([10, 10], minval=0, maxval=10, dtype=tf.int32),
    )
).batch(2)

# Compile and fit the model
model.compile()
try:
    model.fit(train_ds.take(1), epochs=1)
except tf.errors.InvalidArgumentError as e:
    print(f"Error: {e}")