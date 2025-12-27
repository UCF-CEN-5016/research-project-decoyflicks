import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Define a simple model with fixed input length (1 token)
model = models.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# Step 2: Tokenize the input text
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(["eu rejects german call to boycott british lamb"])
sequences = tokenizer.texts_to_sequences(["eu rejects german call to boycott british lamb"])
sample_input = sequences[0]  # This is a sequence of 9 tokens

# Step 3: Save the model
model.save('model.h5')

# Step 4: Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Step 5: Load and run inference with the TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Attempt to input a sequence of 9 tokens, which will cause a dimension mismatch
input_data = np.expand_dims(sample_input, axis=0)  # Shape: (1, 9)
interpreter.set_tensor(input_details['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details['index'])