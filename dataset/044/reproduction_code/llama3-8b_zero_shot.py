import torch
from transformers import JvNTTSForConditionalGeneration, JvNTTSPipeline

model = JvNTTSForConditionalGeneration.from_pretrained('jvn')
pipeline = JvNTTSPipeline(model)

input_text = 'これはテストの文です'
input_ids = ['5418', '1114', '2345', '5419']
output_texts = pipeline(input_text, input_ids=input_ids, max_length=50, early_stopping=True)

print(output_texts)