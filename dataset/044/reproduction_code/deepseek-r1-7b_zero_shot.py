import logging
import torch
from model import load_model, GriffinEllipse
from preprocess import getOOVTokenMapping

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Get OOV token mapping from disk
oov_map = getOOVTokenMapping()

# Load the model
model, _, _ = load_model()

# Test input containing an out-of-vocabulary token
test_input = "I like to eat sushi and rice."

# Filter out of vocabulary tokens
filtered_input = oov_map(test_input)

# Use text to speech synthesis
synthesis = GriffinEllipse(model).synthesize(filtered_input)