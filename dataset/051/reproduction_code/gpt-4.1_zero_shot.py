import tensorflow as tf
import keras_nlp

sampler = keras_nlp.samplers.GreedySampler()
tokens = tf.constant([[1, 2, 3]])
# The following call triggers the error:
sampler(next_token_scores=None,  # dummy placeholder
        tokens=tokens,
        end_token_id=0)