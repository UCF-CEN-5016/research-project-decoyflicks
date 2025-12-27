import tensorflow as tf
from tensorflow import keras
import numpy as np

# Mock NER model with transformer layer (simplified from original example)
def create_mock_ner_model():
    inputs = keras.Input(shape=(None,), dtype='int64')  # Variable length input
    embedding = keras.layers.Embedding(10000, 64)(inputs)
    transformer = keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)(embedding, embedding)
    outputs = keras.layers.Dense(16, activation='softmax')(transformer)
    return keras.Model(inputs, outputs)

# Custom loss to match original example
class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return y_pred  # Simplified for reproduction

# Create and save model
model = create_mock_ner_model()
model.compile(loss=CustomNonPaddingTokenLoss())
model.save('ner_model')

# Conversion to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('ner_model')
tflite_model = converter.convert()

# Save and load TFLite model
with open('ner_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='ner_model.tflite')
interpreter.allocate_tensors()

# Test prediction (simulating tokenized input)
sample_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)  # Length 9 sequence
input_details = interpreter.get_input_details()[0]
input_index = input_details["index"]

# Ensure input shape matches model requirements
input_shape = input_details["shape"]
sample_input = np.expand_dims(sample_input, axis=0)
if input_shape != sample_input.shape:
    raise ValueError(f"Input shape mismatch. Expected: {input_shape}, Actual: {sample_input.shape}")

interpreter.set_tensor(input_index, sample_input)
interpreter.invoke()

# Get the model predictions
output_details = interpreter.get_output_details()[0]
output_data = interpreter.get_tensor(output_details['index'])
print("Model predictions:", output_data)