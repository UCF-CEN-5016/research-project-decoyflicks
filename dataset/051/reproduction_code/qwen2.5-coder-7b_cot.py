import keras_nlp
from keras_nlp import text_generation

def _create_text_generator(start_tokens, end_token_id=3, max_tokens=50):
    return text_generation.TextGenerator(
        start_tokens=start_tokens,
        end_token_id=end_token_id,
        max_tokens=max_tokens,
    )

def _prepare_model_inputs(tokenized_text):
    return [
        [
            [keras_nlp.data.START_TOKEN],
            tokenized_text,
        ],
    ]

def decode_sequences(input_sentences, tokenizer):
    for text in input_sentences:
        tokenized_text = keras_nlp.texts.preprocess_text(text, task='translation')
        model_inputs = _prepare_model_inputs(tokenized_text)
        generator = _create_text_generator(start_tokens=tokenized_text)
        output_tokens = generator(model_inputs)
        translation = tokenizer.detokenize(output_tokens).numpy().lower()
        yield translati