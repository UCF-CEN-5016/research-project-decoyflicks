import tensorflow as tf
import numpy as np

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1, 1))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def save_model_to_tflite(model, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)

def load_tflite_model(filename):
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    return interpreter

def prepare_input_data(input_data):
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Define and train the model
model = create_model()

# Save the model to TFLite
save_model_to_tflite(model, 'model.tflite')

# Load the TFLite model and run inference
interpreter = load_tflite_model('model.tflite')

# Prepare input data
input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
input_data = prepare_input_data(input_data)

# Run inference
output_data = run_inference(interpreter, input_data)