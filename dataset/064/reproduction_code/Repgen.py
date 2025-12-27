import numpy as np
import tensorflow as tf

# Mock custom loss class
class CustomNonPaddingTokenLoss:
    def __call__(self, y_true, y_pred):
        return 0.0

# Create a simple model that expects [1,1] input shape
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.int64),
    tf.keras.layers.Dense(1)
])

# Save the model
model.save('test_model')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('test_model')
tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Mock tokenizer function that returns wrong shape
def tokenize_and_convert_to_ids(text):
    # Returns shape [9] instead of expected [1,1]
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

try:
    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create input with wrong shape
    sample_input = tokenize_and_convert_to_ids("test text")
    sample_input = np.expand_dims(sample_input, axis=0)  # Shape becomes [1,9] instead of [1,1]
    
    # This will raise the dimension mismatch error
    interpreter.set_tensor(input_details[0]['index'], sample_input)
    
except ValueError as e:
    print("Expected ValueError occurred:")
    print(e)
    print("\nThis error occurs because the input shape is [1,9] but model expects [1,1]")