import torch
from mmspeech.models import JVNModel
from mmspeech.text_utils import convert_text_to_sequence, convert_sequence_to_text

# Load the JVN model
model_path = 'jvn.tar.gz'  # replace with your downloaded model path
model = JVNModel(model_path)

# Set up input text and parameters for inference
text = "この文書は、自然言語処理の最新技術について説明します"  # Japanese text
params = {'length_penalty': 0.5, 'beam_width': 10}

# Perform inference to generate text
sequence = convert_text_to_sequence(text)
output_sequence = model.inference(sequence, params)

# Convert the generated sequence back to text
text_after_filtering = convert_sequence_to_text(output_sequence)

print(text_after_filtering)  # This should print the original text, but it's empty instead