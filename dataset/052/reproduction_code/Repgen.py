import tensorflow as tf
import numpy as np
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, GlobalAveragePooling1D, Input
from keras.models import Model

# Define the vocabulary size and number of tags
vocabulary_size = 20000
num_tags = 10

# Load the training dataset from './data/conll_train.txt'
train_data = tf.data.TextLineDataset('./data/conll_train.txt')
for line in train_data.take(5):
    print(line.numpy().decode('utf-8'))

# Tokenize and convert to lowercase, then apply string lookup layer
def preprocess_text(text):
    return text.lower()

tokenized_data = train_data.map(preprocess_text)

# Create batch size of 32 using padded_batch
batch_size = 32
train_dataset = tokenized_data.padded_batch(batch_size)

# Define the NERModel
embedding_dim = 32

inputs = Input(shape=(None,), dtype="string")
x = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)(inputs)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(num_tags, activation="softmax")(x)

ner_model = Model(inputs=inputs, outputs=outputs)

# Compile the model
optimizer = tf.keras.optimizers.Adam()
loss_fn = "sparse_categorical_crossentropy"
metrics = ["accuracy"]
ner_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Train the model for a single epoch
history = ner_model.fit(train_dataset, epochs=1)

# Create sample input text and convert to IDs with lowercase conversion
sample_text = 'eu rejects german call to boycott british lamb'
sample_tokens = [token for token in sample_text.split()]
sample_ids = [vocabulary_size - 1 if token not in vocabulary else vocabulary.index(token) for token in sample_tokens]

# Reshape the sample input to a shape of [1, -1]
sample_input = tf.expand_dims(sample_ids, axis=0)

# Call the ner_model.predict function on the reshaped sample input
predictions = ner_model.predict(sample_input)
print(predictions)