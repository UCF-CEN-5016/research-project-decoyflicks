import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_nlp
import nltk
import numpy as np

MAX_SEQUENCE_LENGTH = 40
TEST_BATCH_SIZE = 1

# Placeholder for tokenizer and model definitions
# These should be defined in your data_preprocessing module
# from data_preprocessing import eng_tokenizer, spa_tokenizer, test_pairs
eng_tokenizer = None  # Placeholder
spa_tokenizer = None  # Placeholder
transformer_model = None  # Placeholder
test_pairs = None  # Placeholder

def decode_sequences(input_sentences):
    encoder_input_tokens = eng_tokenizer(input_sentences)
    encoder_input_tokens = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_tokens, maxlen=MAX_SEQUENCE_LENGTH)
    encoder_input_tokens = encoder_input_tokens[:, :MAX_SEQUENCE_LENGTH]

    def next_fn(prompt, cache, index):
        logits = transformer_model(encoder_input_tokens, prompt)
        logits = logits[:, index - 1, :]
        hidden_states = None
        print(logits.shape, cache.shape)  # Debugging output to trace shapes
        return logits, (), hidden_states

    prompt = tf.constant([[1] + [0] * (MAX_SEQUENCE_LENGTH - 1)])  # Start token
    sampler = keras_nlp.samplers.GreedySampler()
    return sampler(next_fn, prompt, end_token_id=2, index=1)  # Assuming end_token_id is defined

# Placeholder for test pairs
# test_pairs should be defined in your data_preprocessing module
# test_eng_texts, test_spa_texts = zip(*test_pairs)
test_eng_texts = []  # Placeholder
test_spa_texts = []  # Placeholder

bleu_score = []
for eng_text in test_eng_texts:
    translated_tensor = decode_sequences(eng_text)
    translated_text = translated_tensor.numpy().decode('utf-8').replace('[PAD]', '').replace('[START]', '').replace('[END]', '').strip()
    translated_words = translated_text.split()
    score = nltk.translate.bleu_score.modified_precision([test_spa_texts[i].split() for i in range(len(test_spa_texts))], translated_words, n=4)
    bleu_score.append(score)

print(np.mean(bleu_score))

# Run the script to monitor for segmentation fault