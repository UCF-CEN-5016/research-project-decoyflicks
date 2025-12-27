import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import keras_nlp
from nltk.translate import bleu_score

MAX_SEQUENCE_LENGTH = 40
TEST_BATCH_SIZE = 1

# Assuming eng_tokenizer, spa_tokenizer, and test_pairs are defined in data_preprocessing
eng_tokenizer = ...  # Placeholder for actual tokenizer
spa_tokenizer = ...  # Placeholder for actual tokenizer
test_pairs = ...     # Placeholder for actual test pairs

# Placeholder for the transformer model and token IDs
transformer_model = ...  # Placeholder for the actual transformer model
start_token_id = ...     # Placeholder for the actual start token ID
padding_token_id = ...   # Placeholder for the actual padding token ID
end_token_id = ...       # Placeholder for the actual end token ID

def decode_sequences(input_sentences):
    encoder_input_tokens = eng_tokenizer(input_sentences)
    encoder_input_tokens = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_tokens, maxlen=MAX_SEQUENCE_LENGTH)
    if len(encoder_input_tokens) > 40:
        encoder_input_tokens = encoder_input_tokens[:40]

    def next_fn(prompt, cache, index):
        logits, hidden_states = transformer_model(encoder_input_tokens, prompt)
        return logits, (), None  # Returning an empty tuple and None to reproduce the bug

    prompt = tf.constant([[start_token_id] + [padding_token_id] * (MAX_SEQUENCE_LENGTH - 1)])
    sampler = keras_nlp.samplers.GreedySampler()
    return sampler(next_fn, prompt, end_token_id, index=1)

test_eng_texts = [pair[0] for pair in test_pairs]
test_spa_texts = [pair[1] for pair in test_pairs]
bleu_score_list = []

for i in range(len(test_pairs)):
    translated = decode_sequences(test_eng_texts[i])
    translated = translated.numpy().decode('utf-8').replace('[PAD]', '').replace('[START]', '').replace('[END]', '').strip()
    translated_words = translated.split()
    score = bleu_score.modified_precision([test_spa_texts[i].split()], translated_words, n=4)
    bleu_score_list.append(score)

print("Mean BLEU Score:", sum(bleu_score_list) / len(bleu_score_list))

# Run the script and monitor for segmentation fault