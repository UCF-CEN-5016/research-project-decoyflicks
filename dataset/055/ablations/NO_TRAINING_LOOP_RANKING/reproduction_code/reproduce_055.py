import tensorflow as tf
import keras
from keras import layers
import numpy as np
import random
from nltk.translate import bleu_score
import keras_nlp  # Import keras_nlp to fix the undefined variable issue

MAX_SEQUENCE_LENGTH = 40
TEST_BATCH_SIZE = 1

# Load tokenizers and test pairs
eng_tokenizer = ...  # Load English tokenizer
spa_tokenizer = ...  # Load Spanish tokenizer
test_pairs = ...  # Load test pairs

def transformer_model():
    # Define the transformer model
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = layers.Embedding(input_dim=10000, output_dim=256)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(10000, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model

def decode_sequences(input_sentences):
    encoder_input_tokens = eng_tokenizer.texts_to_sequences(input_sentences)
    encoder_input_tokens = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_tokens, maxlen=MAX_SEQUENCE_LENGTH)
    
    def next_fn(prompt, cache, index):
        logits = transformer_model()(encoder_input_tokens)
        return logits, None, cache

    prompt = tf.constant([[1] + [0] * (MAX_SEQUENCE_LENGTH - 1)])  # Start token followed by padding
    sampler = keras_nlp.samplers.GreedySampler(next_fn, prompt, end_token_id=2, index=1)
    
    logits, hidden_states, cache = next(sampler)
    print(logits.shape, cache)  # Log shapes for debugging

    return logits  # Return logits for further processing

for eng_sentence, spa_sentence in test_pairs:
    translated_logits = decode_sequences([eng_sentence])
    translated_sentence = spa_tokenizer.sequences_to_texts(translated_logits.numpy())
    translated_sentence = translated_sentence[0].replace('<start>', '').replace('<end>', '').strip()
    bleu = bleu_score.sentence_bleu([spa_sentence.split()], translated_sentence.split())
    print(f'Translated: {translated_sentence}, BLEU score: {bleu}')