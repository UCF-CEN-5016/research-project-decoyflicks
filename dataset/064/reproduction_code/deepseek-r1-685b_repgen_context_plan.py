import tensorflow as tf
from tensorflow import keras
import numpy as np

# Mock NER model with transformer layer
def create_mock_ner_model():
    inputs = keras.Input(shape=(None,), dtype='int64')  # Variable length input
    embedding = keras.layers.Embedding(10000, 64)(inputs)
    transformer = keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)(embedding, embedding)
    outputs = keras.layers.Dense(16, activation='softmax')(transformer)
    return keras.Model(inputs, outputs)

# Custom loss to match original example
class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return y_pred

# Create and save model
model = create_mock_ner_model()
model.compile(loss=CustomNonPaddingTokenLoss())
model.save('ner_model')

# Conversion to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('ner_model')
tflite_model = converter.convert()

# Save TFLite model to file
with open('ner_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='ner_model.tflite')
interpreter.allocate_tensors()

# Test prediction (simulating tokenized input)
sample_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)  # Length 9 sequence
input_index = interpreter.get_input_details()[0]["index"]

# Error handling for input shape mismatch
try:
    interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))
except ValueError as e:
    print(f"Error: {e}")
    expected_shape = interpreter.get_input_details()[0]["shape"]
    actual_shape = np.expand_dims(sample_input, axis=0).shape
    print(f"Expected input shape: {expected_shape}, Actual input shape: {actual_shape}")