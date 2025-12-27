import keras_nlp
from keras_nlp import text_generation

def decode_sequences(input_sentences):
    for sentence in input_sentences:
        tokenized = keras_nlp.texts.preprocess_text(sentence, task='translation')
        inputs = [
            [
                [keras_nlp.data.START_TOKEN],
                tokenized,
            ],
        ]
        
        # Create a TextGenerator instance
        text_generator = text_generation.TextGenerator(
            start_tokens=tokenized,
            end_token_id=3,  # Assuming this is the correct ID for English to Spanish
            max_tokens=50,
        )
        
        generated_tokens = text_generator(inputs)
        translated = tokenizer.detokenize(generated_tokens).numpy().lower()
        yield translated