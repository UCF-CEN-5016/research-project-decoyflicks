import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.lite.python.convert import TFLiteConverter

# Set up the minimal environment
model_path = 'path_to_your_model.h5'  # Replace with your model path
tflite_model_path = 'path_to_your_tflite_model.tflite'

# Load the Keras model
new_model = keras.models.load_model(model_path, custom_objects={'CustomNonPaddingTokenLoss': CustomNonPaddingTokenLoss()}, compile=False)

# Convert the model to TFLITE
converter = TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()

# Save the TFLITE model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Load the saved TFLITE model and create an interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Define a sample input (e.g., a sequence of IDs)
sample_input = np.array([[1, 2, 3, 4, 5]])  # Replace with your own input data

# Set the input tensor
input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))

# Run the model inference
interpreter.invoke()

# Get the output tensor
output_index = interpreter.get_output_details()[0]["index"]
prediction_mask = interpreter.get_tensor(output_index)

try:
    # Try to set the output tensor (this will trigger the bug)
    interpreter.set_tensor(output_index, prediction_mask)
except ValueError as e:
    print(f"Error: {e}")