import numpy as np
import tensorflow as tf
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "dbmdz/bert-wwm-cased"
MAX_LENGTH = 128

LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

train_dataset = ...
mapping = ...

def tokenize_and_convert_to_ids(text, tk=tokenizer, max_length=MAX_LENGTH):
    tokens = text.split()
    return tk.encode(tokens, return_tensors="tf", max_length=max_length, truncation=True)

sample_text = "eu rejects german call to boycott british lamb"
sample_input = tokenize_and_convert_to_ids(sample_text)
sample_input = tf.reshape(sample_input, shape=[1, -1])

model.compile(optimizer="adam", loss=LOSS)
model.fit(train_dataset, epochs=10)

output = model.predict(sample_input)
prediction_ids = np.argmax(output, axis=-1)[0]
prediction_labels = [mapping[i] for i in prediction_ids]

print(prediction_labels)