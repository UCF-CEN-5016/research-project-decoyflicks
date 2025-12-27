import tensorflow as tf
from tensorflow import keras
import keras_nlp

# Mock tokenizer and model setup (simplified)
vocab_size = 1000
seq_length = 40
embed_dim = 256

# Create a mock model (similar to what might be in the original code)
inputs = keras.Input(shape=(seq_length,), dtype="int32")
x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)
outputs = keras.layers.Dense(vocab_size)(x)
model = keras.Model(inputs, outputs)

# Mock tokenizer (simulating English-to-Spanish)
def mock_tokenizer(texts):
    # Intentionally return inconsistent shapes to trigger the bug
    return tf.ragged.constant([[1, 2, 3], [1, 2, 3, 4, 5]])  # Uneven lengths

# Greedy sampler setup
greedy_sampler = keras_nlp.samplers.GreedySampler()

# Broken decode_sequences function (similar to the bug report)
def decode_sequences(input_sentences):
    encoder_input_tokens = mock_tokenizer(input_sentences).to_tensor()
    
    # Missing padding check → shape mismatch
    def next_fn(prompt, cache, index):
        # This will fail because prompt shape doesn't match model expectations
        logits = model([encoder_input_tokens, prompt])
        return tf.argmax(logits, axis=-1)
    
    # Trigger the segmentation fault
    initial_prompt = tf.constant([[1]])  # Wrong shape
    greedy_sampler.next_step(next_fn, initial_state=[encoder_input_tokens, initial_prompt])

# Trigger the bug
decode_sequences(["hello world", "this should fail"])