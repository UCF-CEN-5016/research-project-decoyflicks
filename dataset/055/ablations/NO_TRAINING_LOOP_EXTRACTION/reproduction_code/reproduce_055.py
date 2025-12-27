import tensorflow as tf
import keras_nlp
import nltk
from data_preprocessing import eng_tokenizer, spa_tokenizer, test_pairs

# Constants for sequence length and batch size
MAX_SEQUENCE_LENGTH = 40
TEST_BATCH_SIZE = 1

# Placeholder variables for model and token IDs
# These should be defined or imported from the appropriate module
model = None  # Define your model here
start_token_id = 1  # Replace with actual start token ID
padding_token_id = 0  # Replace with actual padding token ID
end_token_id = 2  # Replace with actual end token ID

def decode_sequences(input_sentences):
    encoder_input_tokens = eng_tokenizer.texts_to_sequences(input_sentences)
    encoder_input_tokens = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_tokens, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Ensure the input tokens do not exceed the maximum sequence length
    if encoder_input_tokens.shape[1] > MAX_SEQUENCE_LENGTH:
        encoder_input_tokens = encoder_input_tokens[:, :MAX_SEQUENCE_LENGTH]

    def next_fn(prompt, cache, index):
        logits = model(encoder_input_tokens, prompt)  # Call the model with the input tokens and prompt
        hidden_states = None  # This is intentionally left as None to reproduce the bug
        return logits, hidden_states, cache

    prompt = tf.constant([[start_token_id] + [padding_token_id] * (MAX_SEQUENCE_LENGTH - 1)], dtype=tf.int32)
    sampler = keras_nlp.samplers.GreedySampler(next_fn, prompt, end_token_id, index=1)

    return sampler

# Prepare test data
test_eng_texts = [pair[0] for pair in test_pairs]
test_spa_texts = [pair[1] for pair in test_pairs]
bleu_scores = []

# Iterate through test pairs to translate and calculate BLEU scores
for i in range(len(test_pairs)):
    translated_output = decode_sequences([test_eng_texts[i]])
    translated_output = translated_output.numpy().tolist()
    translated_output = [word for word in translated_output if word not in [start_token_id, end_token_id]]
    bleu_score = nltk.translate.bleu_score.modified_precision([test_spa_texts[i]], translated_output, n=4)
    bleu_scores.append(bleu_score)

# Print the mean BLEU score
print("Mean BLEU Score:", sum(bleu_scores) / len(bleu_scores))