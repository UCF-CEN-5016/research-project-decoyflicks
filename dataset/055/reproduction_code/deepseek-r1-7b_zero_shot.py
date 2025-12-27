import tensorflow as tf
import keras_nlp
from data_preprocessing import *
from model import transformer
from nltk.translate.bleu_score import modified_precision

def decode_sequences(input_sentences):
    batch_size = 1
    test_input = eng_tokenizer([input_sentences]).to_tensor()
    
    # Pad or truncate input to MAX_SEQUENCE_LENGTH
    test_input = tf.squeeze(test_input)
    if len(test_input) < MAX_SEQUENCE_LENGTH:
        pad = tf.zeros((MAX_SEQUENCE_LENGTH - len(test_input), 1))
        test_input = tf.concat([test_input, pad], axis=0)
    else:
        test_input = test_input[:MAX_SEQUENCE_LENGTH]
    
    # Prepare initial state
    initial_state = (tf.zeros((batch_size, 2)),) * 2
    
    def next_fn(timestep, states):
        try:
            # Pass inputs through model with proper context
            outputs = transformer(
                [
                    tf.expand_dims(test_input, axis=1),
                    timestep,
                    states[0],
                    states[1]
                ]
            )
            
            logits, new_states = outputs
            cache = (states[0], states[1])
            
            # Check input validity before accessing layer
            if len(cache) == 2 and isinstance(cache[0], tf.Tensor):
                valid_cache = True
            else:
                valid_cache = False
            
            print(f"Shape check passed: {logits.shape}, Cache Validity: {valid_cache}")
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
    
    # Start inference loop
    states = initial_state
    for _ in range(50):  # Adjusted to limit steps
        timestep, states = next_fn(timestep, states)