import tensorflow as tf
import keras_nlp
import numpy as np
from nltk.translate.bleu_score import modified_precision

class EnglishTokenizer:
    def __init__(self):
        self.token_to_id = {'hello': 1, 'world': 2, '[START]': 3, '[END]': 4, '[PAD]': 0}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def __call__(self, sentence):
        tokens = sentence.split(' ')
        token_ids = [self.token_to_id.get(token, 0) for token in tokens]
        return tf.convert_to_tensor(token_ids)

class SpanishTokenizer:
    def __init__(self):
        self.token_to_id = {'hola': 1, 'mundo': 2, '[START]': 3, '[END]': 4, '[PAD]': 0}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def __call__(self, sentence):
        tokens = sentence.split(' ')
        token_ids = [self.token_to_id.get(token, 0) for token in tokens]
        return tf.convert_to_tensor(token_ids)

    def detokenize(self, token_ids):
        tokens = [self.id_to_token.get(id, '') for id in token_ids]
        return ' '.join(tokens)

class Transformer(tf.keras.Model):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = tf.keras.layers.Embedding(5, 256)
        self.decoder = tf.keras.layers.Embedding(5, 256)
        self.transformer = tf.keras.layers.Transformer()

    def call(self, inputs):
        encoder_input, decoder_input = inputs
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input)
        output = self.transformer([encoder_output, decoder_output])
        return output

def decode_sequences(input_sentences, eng_tokenizer, spa_tokenizer, transformer):
    batch_size = 1
    encoder_input_tokens = tf.convert_to_tensor(eng_tokenizer(input_sentences[0]))
    if len(encoder_input_tokens) < 40:
        pads = tf.fill((40 - len(encoder_input_tokens),), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], 0)

    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])
        logits = logits[:, index - 1, :]
        hidden_states = None
        return logits, hidden_states, cache

    start = tf.fill((1, 1), spa_tokenizer.token_to_id('[START]'))
    pad = tf.fill((1, 39), spa_tokenizer.token_to_id('[PAD]'))
    prompt = tf.concat([start, pad], axis=-1)

    generated_tokens = keras_nlp.samplers.GreedySampler()(next_fn, prompt, end_token_id=spa_tokenizer.token_to_id('[END]'), index=1)
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences

def main():
    eng_tokenizer = EnglishTokenizer()
    spa_tokenizer = SpanishTokenizer()
    transformer = Transformer()
    input_sentences = ['hello world']
    translated = decode_sequences(input_sentences, eng_tokenizer, spa_tokenizer, transformer)
    print(translated)

if __name__ == '__main__':
    main()