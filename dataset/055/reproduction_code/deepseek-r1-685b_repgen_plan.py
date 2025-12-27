import tensorflow as tf
import keras_nlp

# Assume transformer, eng_tokenizer, spa_tokenizer, MAX_SEQUENCE_LENGTH are defined as in your example

BATCH_SIZE = 1
MAX_LEN = 40  # max decoding length

def decode_sequences(input_sentences):
    # Tokenize input
    encoder_input_tokens = tf.convert_to_tensor(eng_tokenizer(input_sentences).to_tensor())
    # Pad or truncate input tokens to MAX_LEN
    seq_len = encoder_input_tokens.shape[-1]
    if seq_len < MAX_LEN:
        pads = tf.fill((BATCH_SIZE, MAX_LEN - seq_len), 0)
        encoder_input_tokens = tf.concat([encoder_input_tokens, pads], axis=1)
    elif seq_len > MAX_LEN:
        encoder_input_tokens = encoder_input_tokens[:, :MAX_LEN]

    initial_cache = transformer.get_initial_cache(BATCH_SIZE) if hasattr(transformer, "get_initial_cache") else None

    def next_fn(prompt, cache, index):
        if cache is None:
            logits, new_cache = transformer([encoder_input_tokens, prompt]), None
        else:
            logits, new_cache = transformer([encoder_input_tokens, prompt], cache=cache)

        # Extract logits for current token index
        logits = logits[:, index, :]
        hidden_states = None  # not used here

        # Return logits, hidden_states, updated cache
        return logits, hidden_states, new_cache

    # Build initial prompt with start token + padding
    start_token_id = spa_tokenizer.token_to_id("[START]")
    pad_token_id = spa_tokenizer.token_to_id("[PAD]")
    start = tf.fill((BATCH_SIZE, 1), start_token_id)
    pad = tf.fill((BATCH_SIZE, MAX_LEN), pad_token_id)
    prompt = tf.concat([start, pad], axis=-1)

    # Run greedy decoding
    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next_fn,
        prompt,
        end_token_id=spa_tokenizer.token_to_id("[EOS]"),
        index=1,
        cache=initial_cache,
    )

    # Detokenize output tokens
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences


# Example usage:
test_sentences = ["The weather is nice today."]
translated = decode_sequences(test_sentences)
print(translated.numpy()[0].decode("utf-8"))