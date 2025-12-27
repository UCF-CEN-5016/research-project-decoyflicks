import tensorflow as tf
import keras_nlp

# Minimal dummy tokenizer replacement
class DummyTokenizer:
    def token_to_id(self, token):
        tokens = {"[START]": 1, "[PAD]": 0, "[END]": 2}
        return tokens.get(token, 3)
    def __call__(self, texts):
        # Return a ragged tensor-like structure with fixed length 40
        batch = len(texts)
        # Each sentence tokenized to length 10 for example
        return tf.ragged.constant([[3]*10]*batch)
    def to_tensor(self):
        # For the ragged returned above, return a dense tensor of shape (batch, 10)
        # Here we simulate to_tensor as identity for simplicity
        return tf.constant([[3]*10])

    def detokenize(self, tokens):
        # Return tokens as string tokens for simplicity
        return tf.constant([["hello", "world"]])

# Dummy transformer model simulating expected behavior
class DummyTransformer(tf.keras.Model):
    def call(self, inputs):
        # inputs: list [encoder_input_tokens, prompt]
        encoder_input, prompt = inputs
        batch_size = tf.shape(prompt)[0]
        seq_len = tf.shape(prompt)[1]
        vocab_size = 50
        # Return logits shape: (batch, seq_len, vocab_size)
        return tf.random.uniform((batch_size, seq_len, vocab_size))

# Setup
eng_tokenizer = DummyTokenizer()
spa_tokenizer = DummyTokenizer()
transformer = DummyTransformer()

TEST_BATCH_SIZE = 1
MAX_SEQUENCE_LENGTH = 40

def decode_sequences(input_sentences):
    batch_size = TEST_BATCH_SIZE

    # Tokenize encoder input tokens
    encoder_input_tokens = tf.convert_to_tensor(eng_tokenizer(input_sentences).to_tensor())

    # Pad if less than max length
    if encoder_input_tokens.shape[-1] < MAX_SEQUENCE_LENGTH:
        pads = tf.fill((batch_size, MAX_SEQUENCE_LENGTH - encoder_input_tokens.shape[-1]), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], axis=1)
    # Truncate if longer than 40
    if encoder_input_tokens.shape[-1] > 40:
        encoder_input_tokens = encoder_input_tokens[:, :40]

    # next_fn returns logits, hidden_states, cache
    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])
        logits = logits[:, index - 1, :]  # (batch_size, vocab_size)
        hidden_states = None  # This None is critical and triggers segfault downstream
        return logits, hidden_states, cache

    length = 40
    start = tf.fill((batch_size, 1), spa_tokenizer.token_to_id("[START]"))
    pad = tf.fill((batch_size, length - 1), spa_tokenizer.token_to_id("[PAD]"))
    prompt = tf.concat((start, pad), axis=-1)

    # This call causes segmentation fault in original bug report
    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next_fn,
        prompt,
        end_token_id=spa_tokenizer.token_to_id("[END]"),
        index=1,
    )
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences

# Run minimal test
input_sentence = ["This is a test sentence."]
try:
    translated = decode_sequences(input_sentence)
    print("Generated:", translated)
except Exception as e:
    print("Error:", e)