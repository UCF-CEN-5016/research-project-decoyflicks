import tensorflow as tf
import keras_nlp
import numpy as np

# Mock tokenizer
class MockTokenizer:
    def __init__(self):
        self.vocab_size = 5000
        self.token_to_id = {"[START]": 1, "[PAD]": 0, "[END]": 2}
    
    def __call__(self, texts):
        return tf.ragged.constant([[1] + [0]*39])  # [START] + 39 [PAD]

# Simple transformer model
def build_model():
    encoder_inputs = tf.keras.layers.Input(shape=(40,))
    decoder_inputs = tf.keras.layers.Input(shape=(40,))
    
    # Shared embedding
    embedding = tf.keras.layers.Embedding(5000, 256)
    encoder_emb = embedding(encoder_inputs)
    decoder_emb = embedding(decoder_inputs)
    
    # Transformer
    transformer = tf.keras.layers.Transformer(
        num_heads=8,
        key_dim=256,
        dropout=0.1,
        activation="relu"
    )
    outputs = transformer(encoder_emb, decoder_emb)
    outputs = tf.keras.layers.Dense(5000)(outputs)
    return tf.keras.Model([encoder_inputs, decoder_inputs], outputs)

# Set up
tokenizer = MockTokenizer()
model = build_model()

# Test input
input_sentence = ["test"]  # Content doesn't matter with mock tokenizer
encoder_input = tokenizer(input_sentence).to_tensor()

def next_fn(prompt, cache, index):
    logits = model([encoder_input, prompt])
    logits = logits[:, index-1, :]
    return logits, None, cache  # None for hidden_states

# Sampling setup
prompt = tf.concat([
    tf.fill((1, 1), tokenizer.token_to_id["[START]"]),
    tf.fill((1, 39), tokenizer.token_to_id["[PAD]"])
], axis=-1)

# This should trigger the segfault
try:
    keras_nlp.samplers.GreedySampler()(
        next_fn,
        prompt,
        end_token_id=tokenizer.token_to_id["[END]"],
        index=1
    )
except Exception as e:
    print(f"Error occurred: {e}")
    print("Try running with TF_ENABLE_ONEDNN_OPTS=0 or on CPU")