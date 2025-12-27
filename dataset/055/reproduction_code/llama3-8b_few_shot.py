import tensorflow as tf
import keras_nlp
import numpy as np
from nltk.translate.bleu_score import modified_precision

# Load the model and checkpoint directory
MODEL_CHECKPOINT_DIR = 'path/to/model/checkpoint'

# Define the decode sequences function
def decode_sequences(input_sentences):
    batch_size = 1
    # Tokenize the encoder input.
    encoder_input_tokens = tf.convert_to_tensor(eng_tokenizer(input_sentences).to_tensor())
    if len(encoder_input_tokens[0]) < MAX_SEQUENCE_LENGTH:
        pads = tf.fill((1, MAX_SEQUENCE_LENGTH - len(encoder_input_tokens[0])), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], 1)

    # Define a function that outputs the next token's probability given the input sequence.
    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])
        logits = logits[:, index - 1, :]
        hidden_states = None
        return logits, hidden_states, cache

    # Build a prompt of length 40 with a start token and padding tokens.
    length = 40
    start = tf.fill((batch_size, 1), spa_tokenizer.token_to_id("[START]"))
    pad = tf.fill((batch_size, length - 1), spa_tokenizer.token_to_id("[PAD]"))
    prompt = tf.concat((start, pad), axis=-1)

    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next_fn,
        prompt,
        end_token_id=spa_tokenizer.token_to_id("[END]"),
        index=1  # Start sampling after start token.
    )
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences

# Define the test pairs and tokenize them
test_eng_texts = [pair[0] for pair in test_pairs]
test_spa_texts = [pair[1] for pair in test_pairs]

bleu_score = []
for i in range(len(test_pairs)):
    input_sentence = test_eng_texts[i]
    translated = decode_sequences([input_sentence])
    translated = translated.numpy()[0].decode("utf-8")
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    translated = translated.split(' ')
    bleu = modified_precision([test_spa_texts[i]], translated, n=4)
    bleu_score.append(bleu)

bleu_score_print = np.array(bleu_score)
print("4-gram BLEU score: %f" % (bleu_score_print.mean()))