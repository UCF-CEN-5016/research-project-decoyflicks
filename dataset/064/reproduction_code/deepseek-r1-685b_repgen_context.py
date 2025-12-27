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

def save_model(model, model_path):
    model.compile(loss=CustomNonPaddingTokenLoss())
    model.save(model_path)

def convert_to_tflite(saved_model_path, tflite_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

def load_and_test_tflite_model(tflite_model_path, sample_input):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_index = interpreter.get_input_details()[0]["index"]
    
    try:
        interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))
    except ValueError as e:
        print(f"Error: {e}")
        print("Expected input shape:", interpreter.get_input_details()[0]["shape"])
        print("Actual input shape:", np.expand_dims(sample_input, axis=0).shape)

# Create and save model
model = create_mock_ner_model()
save_model(model, 'ner_model')

# Conversion to TFLite
convert_to_tflite('ner_model', 'ner_model.tflite')

# Test prediction (simulating tokenized input)
sample_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)  # Length 9 sequence
load_and_test_tflite_model('ner_model.tflite', sample_input)