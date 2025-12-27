import tensorflow as tf
import keras_nlp
import numpy as np

vocab = ["[PAD]", "[START]", "[END]", "hello", "world", "!"]
token_to_id = {t: i for i, t in enumerate(vocab)}

class DummyTokenizer:
    def __init__(self):
        self.token_to_id = token_to_id
    def __call__(self, texts):
        ids = [[self.token_to_id.get(t,0) for t in text.split()] for text in texts]
        max_len = max(len(x) for x in ids)
        padded = [x + [0]*(max_len-len(x)) for x in ids]
        return tf.ragged.constant(padded)
    def to_tensor(self):
        return tf.ragged.constant
    def detokenize(self, tokens):
        def detok(seq):
            return " ".join(vocab[t] for t in seq if t < len(vocab))
        return np.array([detok(seq) for seq in tokens.numpy()])

eng_tokenizer = DummyTokenizer()
spa_tokenizer = DummyTokenizer()

MAX_SEQUENCE_LENGTH = 40
TEST_BATCH_SIZE = 1

# Simple transformer-like model with fixed outputs to trigger the bug
input_encoder = tf.keras.Input(shape=(None,), dtype=tf.int32)
input_decoder = tf.keras.Input(shape=(None,), dtype=tf.int32)
embedding = tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=16)
enc_emb = embedding(input_encoder)
dec_emb = embedding(input_decoder)
cross_att = keras_nlp.layers.CachedMultiHeadAttention(
    num_heads=2, key_dim=8, use_causal_mask=True)
x = cross_att(dec_emb, enc_emb)
x = tf.keras.layers.Dense(len(vocab))(x)
transformer = tf.keras.Model([input_encoder, input_decoder], x)

def decode_sequences(input_sentences):
    batch_size = TEST_BATCH_SIZE
    encoder_input_tokens = tf.convert_to_tensor(eng_tokenizer(input_sentences).to_tensor())
    if len(encoder_input_tokens[0]) < MAX_SEQUENCE_LENGTH:
        pads = tf.fill((1, MAX_SEQUENCE_LENGTH - len(encoder_input_tokens[0])), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], 1)
    if encoder_input_tokens.shape[-1] > 40:
        encoder_input_tokens = encoder_input_tokens[:, :40]
    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])
        logits = logits[:, index - 1, :]
        hidden_states = None
        return logits, hidden_states, cache
    length = 40
    start = tf.fill((batch_size, 1), spa_tokenizer.token_to_id["[START]"])
    pad = tf.fill((batch_size, length - 1), spa_tokenizer.token_to_id["[PAD]"])
    prompt = tf.concat((start, pad), axis=-1)
    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next_fn,
        prompt,
        end_token_id=spa_tokenizer.token_to_id["[END]"],
        index=1,
    )
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences

test_eng_texts = ["hello world !"]
test_spa_texts = ["hello world !"]

for input_sentence in test_eng_texts:
    translated = decode_sequences([input_sentence])
    print(translated)