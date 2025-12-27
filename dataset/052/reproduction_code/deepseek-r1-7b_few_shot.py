import tensorflow as tf
from transformers import (
    PreTrainedModel,
    TFBertModel,
)
from tokenization-transformers import tokenize_and_convert_to_ids

# Define a minimal Keras model that includes an incompatible operation
class MinimalNERModel(tf.keras.Model):
    def build(self, vocab_size, max_len):
        self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.vocab_size = vocab_size
        
        # This conditionally adds a custom operation in graph mode
        if True:  # Using eager context for this example
            pass
        else:
            self.attention_layer = tf.keras.layers.Attention()
    
    def call(self, inputs):
        x = self.bert(inputs['input_ids'])
        return tf.math.argmax(x, axis=-1)

# Example token conversion (simulating tokenize_and_convert_to_ids)
def minimal_tokenizer(text):
    tokens = text.split()
    # Simulate a conditionally applied operation
    if len(tokens) > 5:
        tokens = tokens[:5]
    return [tf.constant([i for i in range(len(tokens))], dtype=tf.int32)]

# Create the model instance and dataset (simplified)
model = MinimalNERModel(30512, 768)  # BERT base parameters
dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform((10, 768))).batch(2)

# Attempt to fit the model (this will trigger the error due to graph vs eager operations)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(dataset, epochs=1)