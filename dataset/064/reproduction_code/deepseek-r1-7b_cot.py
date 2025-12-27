import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(32),
    layers.Dense(10)
])

converter = tf.lite.TFLiteConverter(
    model_signatures=model.signatures['default'],
    input_shapes={'input_1': (None, 784)},
    output_shapes={'dense_1': (None, 32), 'dense_2': (None, 10)}
)

tflite_model = converter.convert()

converter = tf.lite.TFLiteConverter.from_saved_model('path_to_my_model')

converter = tf.lite.TFLiteConverter(
    model_signatures=model.signatures,
    input_shapes={input_layer.name: (1, -1)}  # Assuming max_seq_len is variable
)
tflite_model = converter.convert()

import tensorflow as tf

# Load the model with include_input_shape=True
converter = tf.lite.TFLiteConverter.from_saved_model(
    'path_to_my_model',
    include_input_shape=True  # Specify this if needed and possible based on TensorFlow version
)

# Convert to TFLite model
tflite_model = converter.convert()

# Save the TFLite model for later use
tflite_model SaveTo('model.tflite')

# Alternatively, when loading an existing TFLite model (if already converted)
interpreter = tf.lite.Interpreter(model_buffer=tflite_model)

# Prepare sample input ensuring correct dimensions
sample_input = ...  # Ensure shape matches expected by the model

# Set the tensors into the interpreter
interpreter.set_tensor(input_index, sample_input)

# Run inference
output_details = interpreter.get_output_details()
interpreter.run()

# Get results from output details
results = []
for i in range(len(output_details)):
    results.append(interpreter.get_tensor(output_details[i]['index']))