import tensorflow as tf
from transformers import TFRobertaForTokenClassification

# Assuming you have loaded your dataset correctly
train_dataset = ...  # Your dataset pipeline here

# Setup model
model = TFRobertaForTokenClassification.from_pretrained('roberta-base')
model.compile(optimizer='adam', loss=model.crf_loss)
 
# Switch to eager execution mode if needed
tf.config.run_functions_eagerly(True)

# Modify the sample inference code (from the bug report)
sample_input = "eu rejects german call to boycott british lamb"
tokens = sample_input.split()
converted_tokens = lowercase_and_convert_to_ids(tokens)
sample_input_tensor = tf.reshape(converted_tokens, shape=[1, -1])

# Ensure prediction uses eager execution
with tf.GradientTape() as tape:
    outputs = model(sample_input_tensor)
    loss_value = outputs.loss

# Continue training without encountering the error
ner_model.fit(train_dataset, epochs=10)