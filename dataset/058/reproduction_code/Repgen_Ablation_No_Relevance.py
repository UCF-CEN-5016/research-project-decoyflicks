import tensorflow as tf
from tensorflow.keras.applications import ResNet50

batch_size = 2
height, width = 128, 128
input_data = tf.random.uniform((batch_size, height, width, 3), minval=0, maxval=256, dtype=tf.float32)

def get_backbone():
    return ResNet50(weights=None, include_top=False)

model = tf.keras.Sequential([
    get_backbone(),
    # Add other layers as needed
])

loss_fn = lambda y_true, y_pred: y_pred / 0

dummy_dataset = (input_data, input_data)

optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
model.compile(optimizer=optimizer, loss=loss_fn)

history = model.fit(dummy_dataset, epochs=1)

print(history.history['loss'])