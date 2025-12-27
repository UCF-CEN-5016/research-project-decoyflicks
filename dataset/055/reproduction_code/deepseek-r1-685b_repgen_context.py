from keras_nlp.samplers import GreedySampler

def generate_text_with_greedy_sampler(model, start_sequence, max_length):
    """
    Generate text using GreedySampler with the given model and starting sequence.
    
    Args:
    model (keras.Model): The text generation model.
    start_sequence (list): The starting sequence for text generation.
    max_length (int): Maximum length of the generated text.
    
    Returns:
    str: The generated text.
    """
    sampler = GreedySampler(model)
    
    # Generate text using the GreedySampler
    generated_sequence = sampler.generate(start_sequence, max_length)
    
    # Convert the generated sequence to text
    generated_text = ' '.join(generated_sequence)
    
    return generated_text

# Example usage
model = load_model('text_generation_model.h5')
start_sequence = ['The', 'sun', 'is']
max_length = 100
generated_text = generate_text_with_greedy_sampler(model, start_sequence, max_length)
print(generated_text)