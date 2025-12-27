import keras_nlp
import keras
import numpy as np

# Verify versions
print("Keras:", keras.__version__)
print("KerasNLP:", keras_nlp.__version__)

# Mock model and inputs for reproduction
class MockModel:
    def __call__(self, inputs):
        # Return random logits for reproduction
        batch_size, seq_length = inputs.shape
        vocab_size = 1000
        return np.random.rand(batch_size, seq_length, vocab_size)

# Create mock components
model = MockModel()
sampler = keras_nlp.samplers.GreedySampler()

# This will reproduce the error
try:
    # Simulate what the translation example does
    output = sampler(
        model,
        prompt=np.array([[1, 2, 3]]),  # Mock prompt
        sequence_length=10,
        end_token_id=2,  # This causes the error
    )
except TypeError as e:
    print("Reproduced error:", e)

generated_tokens = keras_nlp.samplers.GreedySampler()(
    next,
    prompt,
    sequence_length=MAX_SEQUENCE_LENGTH,
    stop_token_ids=[end_token_id],  # Changed from end_token_id
)