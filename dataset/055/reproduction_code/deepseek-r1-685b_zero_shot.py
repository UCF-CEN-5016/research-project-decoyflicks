import tensorflow as tf
import keras_nlp
import numpy as np

MAX_SEQUENCE_LENGTH = 40
VOCAB_SIZE = 5000

eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=["[PAD]", "[UNK]", "[START]", "[END]"] + [str(i) for i in range(VOCAB_SIZE - 4)],
    lowercase=False
)
spa_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=["[PAD]", "[UNK]", "[START]", "[END]"] + [str(i) for i in range(VOCAB_SIZE - 4)],
    lowercase=False
)

def build_transformer():
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    
    encoder = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=64, num_heads=2, dropout=0.1
    )
    encoder_outputs = encoder(encoder_inputs)
    
    decoder = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=64, num_heads=2, dropout=0.1
    )
    decoder_outputs = decoder(decoder_inputs, encoder_outputs)
    
    outputs = tf.keras.layers.Dense(VOCAB_SIZE)(decoder_outputs)
    return tf.keras.Model([encoder_inputs, decoder_inputs], outputs)

transformer = build_transformer()

def decode_sequences(input_sentences):
    encoder_input_tokens = eng_tokenizer(input_sentences).to_tensor()
    if encoder_input_tokens.shape[-1] > MAX_SEQUENCE_LENGTH:
        encoder_input_tokens = encoder_input_tokens[:, :MAX_SEQUENCE_LENGTH]

    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])
        logits = logits[:, index - 1, :]
        return logits, None, cache

    prompt = tf.concat([
        tf.fill((1, 1), spa_tokenizer.token_to_id("[START]")),
        tf.fill((1, MAX_SEQUENCE_LENGTH - 1), spa_tokenizer.token_to_id("[PAD]"))
    ], axis=-1)

    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next_fn,
        prompt,
        end_token_id=spa_tokenizer.token_to_id("[END]"),
        index=1,
    )
    return generated_tokens

input_sentence = ["test sentence"]
decode_sequences(input_sentence)