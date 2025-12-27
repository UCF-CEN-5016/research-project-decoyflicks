import numpy as np
from keras_nlp.samplers import GreedySampler

# Load pre-trained model and tokenizer (same as in the tutorial)
model, tokenizer = load_pretrained_model_and_tokenizer()

# Sample input sentence
input_sentence = "Hello"

# Decode sequences using the loaded model
def decode_sequences(input_sentences):
    # Define prompt (start token) and padding
    start = tokenizer.encode(" ", return_tensors="tf").numpy()[0]
    pad = np.zeros_like(start)
    
    # Initialize generated tokens list
    generated_tokens = []

    # Call GreedySampler with unexpected keyword argument 'end_token_id'
    greedy_sampler = GreedySampler()
    for sentence in input_sentences:
        prompt = ops.concatenate((start, pad), axis=-1)
        generated_tokens.append(greedy_sampler(end_token_id=42)(next, prompt))
    
    return generated_tokens

# Run decoding
translated = decode_sequences([input_sentence])
print(translated)