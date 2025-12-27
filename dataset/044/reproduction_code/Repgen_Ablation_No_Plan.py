import fairseq.modules.quantization.pq.pq as pq

# Example input text
input_text = "your_input_text_here"

# Assuming the TTS model and necessary components are already set up
# Replace this with actual initialization code if needed
tts_model = ...

# Run inference script
output = tts_model.infer(input_text)

# Print the output
print("text after filtering OOV:", output.text)