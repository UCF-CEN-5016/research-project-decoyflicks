import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure only one GPU is used

# Configuration parameters
batch_size = 32
num_epochs = 10
steps_per_epoch = 1000
 vocab_size = 50000
max_length = 40

# Create sample input data (replace with actual data)
def create_sample_data(batch_size, max_length):
    encoder_input = tf.random.uniform((batch_size, max_length), minval=1, maxval=vocab_size, dtype=tf.int32)
    decoder_input = tf.random.uniform((batch_size, max_length), minval=1, maxval=vocab_size, dtype=tf.int32)
    return encoder_input, decoder_input

encoder_input, decoder_input = create_sample_data(batch_size, max_length)

# Define the Transformer model
class Transformer(Model):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, dff):
        super(Transformer, self).__init__()
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        
    def call(self, inputs):
        # Simplified implementation
        pass

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, dff)

    def call(self, x):
        # Simplified implementation
        pass

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadSelfAttention(d_model, num_heads)
        self.mha2 = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, dff)

    def call(self, x):
        # Simplified implementation
        pass

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.wq = layers.Dense(d_model, activation='relu')
        self.wk = layers.Dense(d_model, activation='relu')
        self.wv = layers.Dense(d_model, activation='relu')

    def call(self, x):
        # Simplified implementation
        pass

class PositionWiseFeedForward(layers.Layer):
    def __init__(self, d_model, dff):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = layers.Dense(dff, activation='relu')
        self.w2 = layers.Dense(d_model)

    def call(self, x):
        # Simplified implementation
        pass

# Initialize the model with correct dimensions
transformer = Transformer(vocab_size=vocab_size, num_layers=6, d_model=max_length, num_heads=8, dff=320)

# Compile the model (including appropriate optimizer and loss function)
transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# During decode_sequences, ensure inputs match expected shape
def decode_sequences(encoder_input):
    # Ensure input tensor dimensions are correct
    if len(encoder_input.shape) != 2 or encoder_input.shape[1] != max_length:
        raise ValueError("Encoder input must have shape (batch_size, max_length)")
        
    # Add batch dimension and pad sequences as needed
    encoded = encoder_input
    return transformer(encoded)

# Run the inference loop with proper checks
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        loss = transformer.train_step(encoder_input, decoder_input)