import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.layers import TextVectorization

# Define constants
BATCH_SIZE = 1
SEQ_LENGTH = 9

# Sample input
input_string = 'eu rejects german call to boycott british lamb'

def tokenize_and_convert_to_ids(input_string):
    # Assuming a TextVectorization layer is already defined and fitted
    vectorization = TextVectorization(max_tokens=10000, output_mode="int", output_sequence_length=SEQ_LENGTH)
    vectorization.adapt(np.array([input_string]))  # Adapt the vectorization layer to the input
    return vectorization(np.array([input_string]))

sample_input = tokenize_and_convert_to_ids(input_string)

# Load the model, assuming CustomNonPaddingTokenLoss is defined elsewhere
# Placeholder for CustomNonPaddingTokenLoss to avoid undefined variable error
class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred)  # Dummy implementation

model = load_model('path_to_my_model', custom_objects={'CustomNonPaddingTokenLoss': CustomNonPaddingTokenLoss()})

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)  # Use from_keras_model for Keras models
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the input tensor with the correct shape
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(sample_input, axis=0))

try:
    interpreter.invoke()
except ValueError as e:
    print(e)  # Expecting: Dimension mismatch. Got 9 but expected 1 for dimension 1 of input 0.