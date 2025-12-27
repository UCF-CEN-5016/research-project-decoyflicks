import torch

input_text = "こんにちは世界"
model_path = "path_to_jvn_model"

# Load the Japanese TTS model
model = torch.load(model_path)

# Initialize tokenizer and encode input text
tokens = model.tokenize(input_text)
encoded_input = model.encode(tokens)

# Call infer function
output = model.infer(encoded_input)

# Check if output is not empty after filtering OOV terms
if len(output['text']) > 0:
    print("Output:", output['text'])
else:
    raise ValueError("Empty output after filtering OOV words")

# Verify the output contains original Japanese characters
assert "こんにちは世界" in output['text']