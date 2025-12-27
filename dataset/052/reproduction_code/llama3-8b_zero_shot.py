import tensorflow as tf
from transformers import AutoModelForTokenClassification, AutoTokenizer

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model_name = "dbmdz/bert-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)

train_dataset = ...

def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    return tokenizer.encode(tokens, return_tensors="tf", max_length=128, truncation=True)

sample_input = tokenize_and_convert_to_ids("eu rejects german call to boycott british lamb")
sample_input = tf.reshape(sample_input, shape=[1, -1])

ner_model.compile(optimizer="adam", loss=loss)
ner_model.fit(train_dataset, epochs=10)

output = ner_model.predict(sample_input)
prediction = np.argmax(output, axis=-1)[0]
prediction = [mapping[i] for i in prediction]

print(prediction)