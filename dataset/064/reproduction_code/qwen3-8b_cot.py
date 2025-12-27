import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense

# Sample data
texts = ["eu rejects german call to boycott british lamb", "another example sentence"]
labels = [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]]

# Tokenize
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Create model with dynamic input shape
input_seq = Input(shape=(None,), dtype='int32')
x = Embedding(input_dim=10000, output_dim=64)(input_seq)
x = GlobalAveragePooling1D()(x)
output = Dense(9, activation='softmax')(x)
model = Model(input_seq, output)

# Compile and save
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.save('my_model.keras')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load and check input shape
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)

# After training, save the model
model.save('my_model.keras')

# Convert to TFLite with fixed input shape
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_types = []
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_type = tf.lite.InferenceType.QUANTIZED_UINT8

# Set fixed input shape
converter.input_type = tf.float32
converter.quantization_aware_training = False

# Save to TFLite
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Pad sequences to max_len
max_len = 10
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Convert to TFLite with fixed input shape
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_types = []
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_type = tf.lite.InferenceType.QUANTIZED_UINT8
converter.input_type = tf.float32
converter.quantization_aware_training = False

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)