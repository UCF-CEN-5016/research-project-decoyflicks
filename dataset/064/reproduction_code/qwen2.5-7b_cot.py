import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["eu rejects german call to boycott british lamb", "another example sentence"]
labels = [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0]]

# Tokenize texts
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to max_len
max_len = 10
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Define and compile model
input_seq = Input(shape=(max_len,), dtype='int32')
x = Embedding(input_dim=10000, output_dim=64)(input_seq)
x = GlobalAveragePooling1D()(x)
output = Dense(9, activation='softmax')(x)
model = Model(input_seq, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the model
model.save('my_model.keras')

# Convert model to TFLite with fixed input shape
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_type = tf.lite.InferenceType.QUANTIZED_UINT8
converter.input_type = tf.float32
converter.quantization_aware_training = False

tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load TFLite model and check input shape
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)