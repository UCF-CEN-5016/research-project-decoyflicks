import tensorflow as tf
import tensorflow_datasets as tfds
import keras_nlp

# Assuming transformer and spa_tokenizer are defined elsewhere in your codebase
# transformer = ...
# spa_tokenizer = ...

MAX_SEQUENCE_LENGTH = 40

def decode_sequences(input_sentences):
    batch_size = 1
    
    # Tokenize the encoder input.
    encoder_input_tokens = tfds.as_dataframe(tfds.load('imdb_reviews', split='train[:5%]'))
    
    if len(encoder_input_tokens[0]) < MAX_SEQUENCE_LENGTH:
        pads = tf.zeros((1, MAX_SEQUENCE_LENGTH - len(encoder_input_tokens[0])))
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], axis=1)
    
    # Define a function that outputs the next token's probability given the input sequence.
    def next(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])[:, index - 1, :]
        hidden_states = None
        return logits, hidden_states
    
    # Build a prompt of length 40 with a start token and padding tokens.
    length = 40
    start = tf.zeros((batch_size, 1))
    pad = tf.zeros((batch_size, length - 1))
    prompt = tf.concat([start, pad], axis=1)
    
    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next,
        prompt,
        index=1,  # Start sampling after start token.
    )
    
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences

# Example usage:
# decoded_sentences = decode_sequences(["I love this movie!"])
# print(decoded_sentences)