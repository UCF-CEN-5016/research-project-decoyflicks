import keras_nlp

def decode_sequences(input_sentences):
    start_tokens = [0]  # Assuming token 0 is used for start
    end_token_id = 0     # End token id, adjust as per model

    prompt = ops.concatenate((start_tokens, pad), axis=-1)

    generated_tokens = (keras_nlp.samplers.GreedySampler(
        start_tokens=start_tokens,
        end_token_id=end_token_id
    ))(prompt)