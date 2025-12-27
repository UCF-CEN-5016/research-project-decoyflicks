import tensorflow as tf
import numpy as np

# Define a model with input shape (1, 1)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1, 1))
])

model.compile(optimizer='adam', loss='mse')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load and run inference
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input has shape (1, 9) but model expects (1, 1)
input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
input_data = np.expand_dims(input_data, axis=0)  # Shape (1,9)

# This will cause the dimension mismatch
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()