import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import keras_nlp
import nltk

MAX_SEQUENCE_LENGTH = 40
TEST_BATCH_SIZE = 1

# Assuming eng_tokenizer, spa_tokenizer, and test_pairs are defined in data_preprocessing
from data_preprocessing import eng_tokenizer, spa_tokenizer, test_pairs
from model import transformer_model

# Define token IDs (these should be set according to your model's requirements)
start_token_id = 1  # Example value, replace with actual start token ID
padding_token_id = 0  # Example value, replace with actual padding token ID
end_token_id = 2  # Example value, replace with actual end token ID

def decode_sequences(input_sentences):
    encoder_input_tokens = eng_tokenizer(input_sentences)
    encoder_input_tokens = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_tokens, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    def next_fn(prompt, cache, index):
        logits, hidden_states, cache = transformer_model(encoder_input_tokens, prompt)
        return logits, hidden_states, cache

    prompt = tf.constant([[start_token_id] + [padding_token_id] * (MAX_SEQUENCE_LENGTH - 1)], dtype=tf.int32)
    sampler = keras_nlp.samplers.GreedySampler()
    sampler(next_fn, prompt, end_token_id, index=1)

test_eng_texts = [pair[0] for pair in test_pairs]
test_spa_texts = [pair[1] for pair in test_pairs]
bleu_score = []

for input_sentence in test_eng_texts:
    output_sentence = decode_sequences(input_sentence)
    output_sentence = output_sentence.replace('<start>', '').replace('<end>', '').strip().split()
    score = nltk.translate.bleu_score.modified_precision(test_spa_texts, output_sentence, n=4)
    bleu_score.append(score)

# Run the script and monitor for segmentation faults