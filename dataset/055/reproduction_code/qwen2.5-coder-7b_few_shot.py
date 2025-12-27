import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.nlp import GreedyDecoder, Tokenizer

# Configuration constants
VOCAB_SIZE = 1000
D_MODEL = 64
NUM_HEADS = 8
MAX_SEQ_LENGTH = 40

class SimpleCrossAttentionModel(Model):
    """A minimal transformer-like model with token & position embeddings and a cross-attention layer."""
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, max_seq_length: int):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = Embedding(max_seq_length, d_model)
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.output_dense = Dense(d_model)

    def call(self, inputs):
        x = self.token_embedding(inputs)
        positions = tf.range(tf.shape(x)[1])
        positions = tf.broadcast_to(positions, tf.shape(x))
        x = x + self.position_embedding(positions)
        x = self.cross_attention(x, x, x)
        return self.output_dense(x)

def generate_empty_logits():
    """Return a tensor of zeros simulating empty logits."""
    return tf.zeros((1, MAX_SEQ_LENGTH, D_MODEL))

def perform_decode(decoder: GreedyDecoder, logits_fn):
    """Run greedy decoding using provided decoder and logits function."""
    logits = logits_fn()
    return decoder(logits, sequence_length=tf.ones((1, MAX_SEQ_LENGTH)))

# Instantiate the model (kept for structural parity with the original code)
model = SimpleCrossAttentionModel(VOCAB_SIZE, D_MODEL, NUM_HEADS, MAX_SEQ_LENGTH)

# Example input (not used further but retained as in original)
example_input = tf.random.uniform(shape=(1, MAX_SEQ_LENGTH, D_MODEL))

# Prepare tokenizer and greedy decoder
tokenizer = Tokenizer(vocabulary_size=VOCAB_SIZE)
greedy_decoder = GreedyDecoder(tokenizer, pad_token=0)

# Attempt to decode and catch potential shape/argument errors
try:
    perform_decode(greedy_decoder, generate_empty_logits)
except tf.errors.InvalidArgumentError as e:
    print("Caught an error during decoding:", e)