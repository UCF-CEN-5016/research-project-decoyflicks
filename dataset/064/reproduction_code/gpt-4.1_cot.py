import tensorflow as tf
from tensorflow import keras
import numpy as np

# Minimal custom loss placeholder (to mimic original environment)
class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_pred - y_true)  # dummy implementation

# Minimal model to mimic NER transformer model input/output shape
def create_minimal_ner_model():
    input_ids = keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    # Simple embedding + dense to mimic token classification output
    x = keras.layers.Embedding(input_dim=1000, output_dim=16)(input_ids)
    x = keras.layers.Dense(5, activation="softmax")(x)  # 5 classes for NER labels
    model = keras.Model(inputs=input_ids, outputs=x)
    return model

# Create and save the minimal model
model = create_minimal_ner_model()
model.compile(loss=CustomNonPaddingTokenLoss())
save_path = "minimal_ner_model"
model.save(save_path, include_optimizer=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
tflite_model = converter.convert()
tflite_path = "minimal_ner_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

# Load the tflite model with Interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

# Check input details - this should give fixed shape (usually batch=1, length=1)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite input details:", input_details)
print("TFLite output details:", output_details)

# Prepare a test input with length > 1 tokens, e.g. 9 tokens
sample_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)

# This triggers the dimension mismatch error because TFLite expects input shape [1,1]
try:
    input_index = input_details[0]["index"]
    # Note: interpreter expects shape [1,1] but we're providing [1,9]
    interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))  # shape (1,9)
    interpreter.invoke()
    output_index = output_details[0]["index"]
    prediction_mask = interpreter.get_tensor(output_index)
    print("Prediction:", prediction_mask)
except ValueError as e:
    print("Caught ValueError:", e)