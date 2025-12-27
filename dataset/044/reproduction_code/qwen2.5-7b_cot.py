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
    filtered = ' '.join(word for word in text.split() if word in vocab)
    return filtered

def main():
    vocab_path = 'vocab.txt'
    input_text = 'こんにちは、世界'

    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('こんにちは ありがとう おはよう')

    vocab = load_vocabulary(vocab_path)
    filtered_text = filter_oov(input_text, vocab)

    print("Text after filtering OOV:", filtered_text)

if __name__ == "__main__":
    main()