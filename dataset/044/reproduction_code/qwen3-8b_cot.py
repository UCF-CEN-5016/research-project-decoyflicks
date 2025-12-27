import os

def load_vocabulary(vocab_path):
    """Load vocabulary from a file."""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = set(f.read().split())
    return vocab

def filter_oov(text, vocab):
    """Filter out-of-vocabulary tokens from the input text."""
    words = text.split()
    filtered = [word for word in words if word in vocab]
    return ' '.join(filtered)

def main():
    # Simulate the model's vocabulary (e.g., Japanese words)
    vocab_path = 'vocab.txt'  # This should be a file containing the model's vocabulary
    input_text = 'こんにちは、世界'  # Example input text (Japanese)

    # Create a minimal vocab file for demonstration
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('こんにちは ありがとう おはよう')  # Only these words are in the vocab

    # Load and filter the input text
    vocab = load_vocabulary(vocab_path)
    filtered_text = filter_oov(input_text, vocab)

    # Output the result
    print("Text after filtering OOV:", filtered_text)

if __name__ == "__main__":
    main()