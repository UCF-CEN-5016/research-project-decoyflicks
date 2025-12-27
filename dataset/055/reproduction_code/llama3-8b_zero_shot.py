import tensorflow as tf
from tensorflow.keras_nlp import samplers
from nltk.translate.bleu_score import modified_precision

TEST_BATCH_SIZE = 1

def decode_sequences(input_sentences):
    batch_size = TEST_BATCH_SIZE

    # Tokenize the encoder input.
    encoder_input_tokens = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    if len(encoder_input_tokens[0]) < 10:
        pads = tf.fill((1, 10 - len(encoder_input_tokens[0])), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], 1)

    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])
        logits = logits[:, index - 1, :]
        hidden_states = None
        return logits, hidden_states, cache

    length = 10
    start = tf.fill((batch_size, 1), 0)
    pad = tf.fill((batch_size, length - 1), 0)
    prompt = tf.concat((start, pad), axis=-1)

    generated_tokens = samplers.GreedySampler()(next_fn, prompt, end_token_id=0, index=1)
    generated_sentences = decoded(generated_tokens)
    return generated_sentences

def decoded(generated_tokens):
    return ''.join(str(x) for x in generated_tokens[0])

bleu_score = []
for i in range(2):
    input_sentence = [1, 2, 3]
    translated = decode_sequences([input_sentence])
    translated = translated.numpy()[0].decode("utf-8")
    bleu = modified_precision([[4, 5, 6]], translated.split(' '), n=4)
    bleu_score.append(bleu)

print ("4-gram BLEU score: %f" % (np.array(bleu_score).mean()))