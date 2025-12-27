import tensorflow as tf
from keras_nlp.models import transformer

# Load dataset and preprocess
train_dataset = ...  # Load English-Spanish translation training data
test_sentences = ...  # Sample test sentences

# Tokenizers
english_tokenizer = ...
spanish_tokenizer = ...

# Model architecture
model = transformer.Transformer(
    num_heads=8,
    embed_dim=512,
    max_length=40,
    vocab_size=25000
)

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_dataset, epochs=1)

# Decode sequences function
def decode_sequences(input_sentences):
    beam_search = keras_nlp.experimental.samplers.BeamSearch(
        model,
        max_length=40,
        start_token_id=spanish_tokenizer.token_to_id("<start>"),
        end_token_id=spanish_tokenizer.token_to_id("<end>"),
        pad_token_id=spanish_tokenizer.token_to_id("<pad>"),
        eos_token_ids=[spanish_tokenizer.token_to_id("<end>")],
        beam_size=4,
        select_top_k=None
    )
    
    translated_sequences = []
    for sentence in input_sentences:
        encoded_input = english_tokenizer.encode(sentence)
        predictions = beam_search(encoded_input)
        decoded_output = spanish_tokenizer.decode(predictions[0])
        translated_sequences.append(decoded_output)
    
    return translated_sequences

# Test function
translated_sentence = decode_sequences(["This is an example."])[0]
print(translated_sentence)