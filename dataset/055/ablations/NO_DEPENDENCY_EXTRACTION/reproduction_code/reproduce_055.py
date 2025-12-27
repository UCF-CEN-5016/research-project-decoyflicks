import tensorflow as tf
from data_preprocessing import eng_tokenizer, spa_tokenizer, test_pairs
from model import transformer
from nltk.translate.bleu_score import modified_precision
import keras_nlp  # Added import for keras_nlp

MAX_SEQUENCE_LENGTH = 40
TEST_BATCH_SIZE = 1

def decode_sequences(input_sentence):
    encoder_input_tokens = eng_tokenizer(input_sentence)
    if len(encoder_input_tokens) < MAX_SEQUENCE_LENGTH:
        encoder_input_tokens += [0] * (MAX_SEQUENCE_LENGTH - len(encoder_input_tokens))
    encoder_input_tokens = encoder_input_tokens[:MAX_SEQUENCE_LENGTH]

    def next_fn(prompt, cache, index):
        logits = transformer(tf.convert_to_tensor([encoder_input_tokens]), prompt)
        logits = logits[:, index - 1, :]
        return logits, None, cache

    prompt = [1] + [0] * (MAX_SEQUENCE_LENGTH - 1)
    sampler = keras_nlp.samplers.GreedySampler()
    return sampler(next_fn, prompt, end_token_id=2, index=1)

test_eng_texts = [pair[0] for pair in test_pairs]
test_spa_texts = [pair[1] for pair in test_pairs]
bleu_score = []

for input_sentence in test_eng_texts:
    translated_sentence = decode_sequences(input_sentence)
    translated_sentence = translated_sentence.numpy().decode('utf-8').replace('[PAD]', '').replace('[START]', '').replace('[END]', '').strip()
    translated_words = translated_sentence.split()
    score = modified_precision(4, [translated_words], [test_spa_texts[test_eng_texts.index(input_sentence)]])
    bleu_score.append(score)

print("Average BLEU score:", sum(bleu_score) / len(bleu_score))