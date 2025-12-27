import tensorflow as tf
from keras_nlp import samplers
from nltk.translate.bleu_score import modified_precision

# Minimal environment setup
TEST_BATCH_SIZE = 1
MAX_SEQUENCE_LENGTH = 40

# Triggering conditions
input_sentence = "Hello, world!"

# Wrap code in a function to isolate the issue
def reproduce_bug():
    # Set up the tokenizer and prompt
    eng_tokenizer = tf.convert_to_tensor(["[START]", "[PAD]"])
    spa_tokenizer = tf.convert_to_tensor(["[END]"])

    # Define the transformer model
    transformer = tf.keras.Sequential([...])

    # Define the next_fn function
    def next_fn(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])
        return logits

    # Build a prompt of length 40 with start token and padding tokens
    prompt = tf.concat((eng_tokenizer, pad), axis=-1)

    # Use the GreedySampler to generate tokens
    generated_tokens = samplers.GreedySampler()(next_fn, prompt, end_token_id=spa_tokenizer.token_to_id("[END]"), index=1)

    # Print the generated tokens
    print(generated_tokens)

# Call the reproduce_bug function to trigger the bug
reproduce_bug()