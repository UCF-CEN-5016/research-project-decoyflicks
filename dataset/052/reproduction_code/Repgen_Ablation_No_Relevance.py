import tensorflow as tf
import numpy as np

batch_size = 32
sequence_length = 100

# Create random integer input data with shape (batch_size, sequence_length)
input_data = tf.random.randint(1, 20000, size=(batch_size, sequence_length))

# Lowercase the tokens using tf.strings.lower function
# Since tokens are integers, we need to convert them to strings first
string_input_data = tf.as_string(input_data)
lowercased_tokens = tf.strings.lower(string_input_data)

# Convert tokens to IDs using a StringLookup layer with predefined vocabulary of size 20000
vocab_size = 20000
lookup_layer = tf.keras.layers.StringLookup(vocabulary=range(1, vocab_size + 1))
input_ids = lookup_layer(lowercased_tokens)

# Reshape the input data to shape (batch_size, -1)
reshaped_input_ids = tf.reshape(input_ids, (batch_size, -1))

# Create a sample NER model with 3 dense layers and output dimension equal to the number of tags
num_tags = 10
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_tags, activation='softmax')
])

# Compile the model with 'adam' optimizer and custom non-padding token loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Fit the model on a dummy dataset created using tf.data.Dataset.from_tensor_slices
dummy_dataset = tf.data.Dataset.from_tensor_slices((reshaped_input_ids, reshaped_input_ids))
history = model.fit(dummy_dataset.batch(batch_size), epochs=1)

# Verify that the model predicts output of shape (batch_size, sequence_length, num_tags)
predictions = model.predict(reshaped_input_ids)
assert predictions.shape == (batch_size, sequence_length, num_tags)

# Assert that the predicted output contains NaN values in at least one element
assert np.isnan(predictions).any()