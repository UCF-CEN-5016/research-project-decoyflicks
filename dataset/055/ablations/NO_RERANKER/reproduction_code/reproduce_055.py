import os
import tensorflow as tf
from tensorflow import keras
import keras_nlp  # Added import for keras_nlp
from data_preprocessing import eng_tokenizer, spa_tokenizer, test_pairs
from model import transformer
from nltk.translate.bleu_score import modified_precision

MAX_SEQUENCE_LENGTH = 40
TEST_BATCH_SIZE = 1

def decode_sequences(input_sentences):
    encoder_input_tokens = eng_tokenizer(input_sentences)
    encoder_input_tokens = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_tokens, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Ensure the input tokens do not exceed the maximum sequence length
    if len(encoder_input_tokens) > MAX_SEQUENCE_LENGTH:
        encoder_input_tokens = encoder_input_tokens[:MAX_SEQUENCE_LENGTH]

    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])
        hidden_states = None  # This will be None as per the bug report
        print(f'Logits shape: {logits.shape}, Cache shape: {cache.shape if cache is not None else "None"}')
        return logits, hidden_states, cache

    prompt = tf.constant([[1] + [0] * (MAX_SEQUENCE_LENGTH - 1)], dtype=tf.int32)
    sampler = keras_nlp.samplers.GreedySampler()
    sampler(next_fn, prompt, end_token_id=2, index=1)

test_sentences = [pair[0] for pair in test_pairs]
bleu_scores = []

for sentence in test_sentences:
    translated = decode_sequences(sentence)
    translated = translated[0].numpy().tolist()
    translated = [word for word in translated if word not in [0, 1, 2]]  # Remove special tokens
    bleu_score = modified_precision(reference=[pair[1].split() for pair in test_pairs], hypothesis=translated, n=4)
    bleu_scores.append(bleu_score)

# Run the script and monitor for segmentation fault