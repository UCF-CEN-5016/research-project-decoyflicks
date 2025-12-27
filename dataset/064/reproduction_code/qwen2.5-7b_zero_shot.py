import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def define_model():
    model = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def tokenize_text(text):
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    return sequences

def save_model(model, filename):
    model.save(filename)

def convert_to_tflite(model, tflite_filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)

def load_and_run_inference(model_filename, tflite_filename, sample_input):
    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_data = np.expand_dims(sample_input, axis=0)
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    return output_data

# Step 1: Define a simple model with fixed input length (1 token)
model = define_model()

# Step 2: Tokenize the input text
sample_text = "eu rejects german call to boycott british lamb"
sample_input = tokenize_text(sample_text)[0]

# Step 3: Save the model
save_model(model, 'model.h5')

# Step 4: Convert the model to TFLite format
convert_to_tflite(model, 'model.tflite')

# Step 5: Load and run inference with the TFLite model
output_data = load_and_run_inference('model.h5', 'model.tflite', sample_input)