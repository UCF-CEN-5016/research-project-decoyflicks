import random
from keras_nlp.samplers import GreedySampler
import numpy as np

class Sampler(GreedySampler):
    def __call__(self, next=None, end_token_id=0, **kwargs):
        pass

start = np.array([1])
pad = np.array([2])

def decode_sequences(input_sentences):
    prompt = ops.concatenate((start, pad), axis=-1)
    generated_tokens = Sampler()(next=None, end_token_id=0)  # This line reproduces the error
    return generated_tokens

ops = None  # Replace with actual ops from keras_nlp

if __name__ == "__main__":
    for i in range(2):
        input_sentence = random.choice(["example", "example"])
        translated = decode_sequences([input_sentence])