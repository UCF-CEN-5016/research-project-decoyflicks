import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

vocab = ["[PAD]", "[UNK]", "eu", "rejects", "german", "call", "to", "boycott", "british", "lamb"]
tags = ["O", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
mapping = {i: tag for i, tag in enumerate(tags)}

def lowercase_and_convert_to_ids(tokens):
    return [vocab.index(token.lower()) if token.lower() in vocab else 1 for token in tokens]

class NERModel(keras.Model):
    def __init__(self, num_tags, vocab_size, embedding_dim=32):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.bilstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))
        self.dense = layers.Dense(num_tags)
        self.softmax = layers.Softmax()

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.bilstm(x)
        x = self.dense(x)
        return self.softmax(x)

ner_model = NERModel(len(tags), len(vocab))
ner_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

train_data = np.array([[2, 3, 4, 5, 6, 7, 8, 9]])
train_labels = np.array([[1, 0, 2, 0, 0, 0, 3, 0]])
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(1)

ner_model.fit(train_dataset, epochs=1)

def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    return lowercase_and_convert_to_ids(tokens)

sample_input = tokenize_and_convert_to_ids("eu rejects german call to boycott british lamb")
sample_input = tf.reshape(sample_input, shape=[1, -1])
output = ner_model.predict(sample_input)