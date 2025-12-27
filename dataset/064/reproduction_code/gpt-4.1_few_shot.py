import numpy as np
import tensorflow as tf
from tensorflow import keras

# Minimal dummy tokenizer function returning token ids (variable length)
def tokenize_and_convert_to_ids(text):
    # Simple whitespace tokenizer mapped to dummy ids
    token_map = {'eu':1, 'rejects':2, 'german':3, 'call':4, 'to':5, 'boycott':6, 'british':7, 'lamb':8}
    return np.array([token_map.get(tok, 0) for tok in text.split()])  # variable length

# Dummy Transformer-based NER model with variable input length
input_ids = keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
embedding_layer = keras.layers.Embedding(input_dim=10, output_dim=4)(input_ids)
x = keras.layers.GlobalAveragePooling1D()(embedding_layer)
output = keras.layers.Dense(5, activation='softmax')(x)
model = keras.Model(inputs=input_ids, outputs=output)

# Save the model (simulate trained model)
model.save('saved_ner_model', include_optimizer=False)

# Convert the saved model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_ner_model')
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load TFLite interpreter and check input details
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model input shape:", input_details[0]['shape'])

# Prepare sample input with variable length tokens (length 9)
sample_input = tokenize_and_convert_to_ids(
    "eu rejects german call to boycott british lamb"
)  # length=9

# Attempt to set input tensor will raise dimension mismatch if input shape is fixed to [1,1]
input_index = input_details[0]['index']

# This line will raise:
# ValueError: Cannot set tensor: Dimension mismatch. Got 9 but expected 1 for dimension 1 of input 0.
interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))

interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
print("Prediction:", prediction)