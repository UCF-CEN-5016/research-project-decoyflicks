import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer

# 1. Set up minimal environment
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_convert_to_ids(text):
    tokens = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_ids(tokens)

# 2. Create a sample model similar to NER transformer
def create_sample_model():
    input_layer = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name="input_ids")
    embedding = tf.keras.layers.Embedding(30522, 128)(input_layer)
    output = tf.keras.layers.Dense(9, activation="softmax")(embedding)
    return tf.keras.Model(inputs=input_layer, outputs=output)

model = create_sample_model()
model.save("temp_ner_model")

# 3. Convert to TFLite (this will trigger the issue)
converter = tf.lite.TFLiteConverter.from_saved_model("temp_ner_model")
tflite_model = converter.convert()

with open("ner_model.tflite", "wb") as f:
    f.write(tflite_model)

# 4. Try to run inference (triggering condition)
interpreter = tf.lite.Interpreter(model_path="ner_model.tflite")
interpreter.allocate_tensors()

sample_input = tokenize_and_convert_to_ids("eu rejects german call to boycott british lamb")
input_index = interpreter.get_input_details()[0]["index"]

# This will raise the dimension mismatch error
interpreter.set_tensor(input_index, np.expand_dims(sample_input, axis=0))

converter = tf.lite.TFLiteConverter.from_saved_model("temp_ner_model")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True

# Set input shape to be dynamic in the first dimension
converter.input_shapes = {0: [1, None]}  # Batch size 1, variable sequence length

tflite_model = converter.convert()