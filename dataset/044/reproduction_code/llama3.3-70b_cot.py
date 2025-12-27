import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Minimal Setup
# Assuming the model is a sequence-to-sequence model and we have a tokenizer for it
model_name = "jvn_model"  # Placeholder for the actual model name from the downloaded tar.gz
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Sample Input Text
input_text = "This is a sample text to test the model."

# Preprocess the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass to get the output
outputs = model.generate(**inputs)

# Decode the output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Simulating OOV filtering
def filter_oov(text, vocab):
    """Simple function to simulate OOV filtering."""
    words = text.split()
    filtered_text = [word for word in words if word in vocab]
    return " ".join(filtered_text)

# Assuming we have a vocabulary list from the model
vocab = set(tokenizer.vocab.keys())

# Apply OOV filtering
filtered_text = filter_oov(decoded_output, vocab)

# This should print the text after filtering OOV
print("Text after filtering OOV:", filtered_text)

# If the output is empty, it suggests the decoded output did not match any words in the vocabulary,
# indicating a potential issue with the model's vocabulary or the decoding process.