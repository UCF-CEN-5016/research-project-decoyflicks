import keras_nlp
import keras
from keras import ops

class MockModel:
    def __call__(self, inputs):
        batch_size = ops.shape(inputs)[0]
        length = ops.shape(inputs)[1]
        return ops.zeros((batch_size, length, 10))

def decode_sequences(input_sentences):
    model = MockModel()
    start = ops.zeros((1, 1), dtype="int32")
    pad = ops.zeros((1, 9), dtype="int32")
    prompt = ops.concatenate((start, pad), axis=-1)
    generated_tokens = keras_nlp.samplers.GreedySampler()(
        model,
        prompt,
        end_token_id=2,
        max_length=10
    )

decode_sequences(["test"])