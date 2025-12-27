import tensorflow as tf
from tensorflow import keras
import keras_nlp
import numpy as np

class SamplerConfig:
    """
    Configuration class for sampler parameters.
    """
    def __init__(self):
        self.vocab_size = 1000
        self.seq_length = 40
        self.embed_dim = 256
        self.max_decode_length = 20 # Max length for generated sequences
        self.start_token_id = 1    # Example start token ID
        self.end_token_id = 0      # Example end token ID (padding token)

class MockSeq2SeqModel(keras.Model):
    """
    A mock sequence-to-sequence model designed to work with KerasNLP samplers.
    It takes encoder outputs (as part of cache) and current decoder prompt.
    """
    def __init__(self, vocab_size, embed_dim, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = keras.layers.Embedding(vocab_size, embed_dim)
        self.decoder_dense = keras.layers.Dense(vocab_size)
        # Simplified "encoder" path for demonstration; actual encoder would be more complex
        self.encoder_embedding = keras.layers.Embedding(vocab_size, embed_dim)
        self.encoder_output_projection = keras.layers.Dense(embed_dim)

    def call(self, inputs, cache=None):
        # inputs[0] is encoder_input_tokens (from cache), inputs[1] is decoder_input_tokens (prompt)
        encoder_input_tokens = inputs[0]
        decoder_input_tokens = inputs[1]

        # Process encoder inputs (simplified to just embedding and projection)
        encoder_outputs = self.encoder_output_projection(self.encoder_embedding(encoder_input_tokens))

        # Process decoder inputs (prompt)
        x = self.token_embedding(decoder_input_tokens)

        # In a real model, this would involve attention to encoder_outputs
        # For this mock, we'll just combine them in a simplified way
        # We'll use the last token of the prompt to predict the next
        last_decoder_token_embedding = x[:, -1, :]
        
        # Simple interaction with encoder_outputs (e.g., average pooling)
        # In a real model, this would be cross-attention
        context_vector = tf.reduce_mean(encoder_outputs, axis=1)

        combined_features = tf.concat([last_decoder_token_embedding, context_vector], axis=-1)
        
        # Project to vocab size for next token prediction
        logits = self.decoder_dense(combined_features)
        return logits

def mock_tokenizer_padded(texts: list[str], seq_length: int, vocab_size: int) -> tf.Tensor:
    """
    Mocks a tokenizer that returns padded tensors.
    """
    tokenized_sequences = []
    for text in texts:
        # Simulate tokenization by assigning random IDs
        # and ensure padding to seq_length
        tokens = np.random.randint(2, vocab_size, size=(np.random.randint(1, seq_length + 1),)).tolist()
        if len(tokens) > seq_length:
            tokens = tokens[:seq_length]
        padded_tokens = tokens + [0] * (seq_length - len(tokens)) # Pad with 0s
        tokenized_sequences.append(padded_tokens)
    return tf.constant(tokenized_sequences, dtype=tf.int32)

def decode_sequences_with_sampler(input_sentences: list[str], config: SamplerConfig):
    """
    Decodes sequences using a KerasNLP sampler with a custom next_fn.
    """
    # Tokenize input sentences and convert to padded tensors
    encoder_input_tokens = mock_tokenizer_padded(input_sentences, config.seq_length, config.vocab_size)
    batch_size = tf.shape(encoder_input_tokens)[0]

    # Initialize the mock sequence-to-sequence model
    model = MockSeq2SeqModel(config.vocab_size, config.embed_dim, config.seq_length)

    # Define the next_fn for the sampler
    def next_fn(prompt, cache, index):
        # cache[0] contains encoder_input_tokens
        encoder_outputs_from_cache = cache[0]
        # model expects [encoder_inputs, decoder_inputs (prompt)]
        logits = model([encoder_outputs_from_cache, prompt])
        return logits

    # Initialize greedy sampler
    greedy_sampler = keras_nlp.samplers.GreedySampler(
        max_length=config.max_decode_length,
        end_token_id=config.end_token_id
    )

    # Initial prompt should be the start token for each sequence in the batch
    initial_prompt = tf.fill((batch_size, 1), config.start_token_id)

    print(f"Starting decoding for batch size: {batch_size}")
    # The initial_cache is passed to the next_fn's 'cache' argument
    generated_tokens = greedy_sampler(
        next_fn,
        prompt=initial_prompt,
        cache=[encoder_input_tokens] # Pass encoder_input_tokens as initial cache
    )
    print("Decoding complete.")
    return generated_tokens

def main():
    config = SamplerConfig()

    input_sentences = ["hello world", "this should succeed", "another example sentence"]

    try:
        generated_sequences = decode_sequences_with_sampler(input_sentences, config)
        print("\nGenerated sequences (token IDs):")
        print(generated_sequences.numpy())
    except Exception as e:
        print(f"\nAn error occurred during decoding: {e}")

if __name__ == "__main__":
    main()
