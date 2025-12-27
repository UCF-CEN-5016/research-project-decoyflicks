import re
from unicodedata import normalize

def preprocess_text(text, lang):
    """Minimal text preprocessing similar to MMS TTS"""
    # Normalize Unicode
    text = normalize("NFKC", text)
    
    # Japanese-specific cleaning
    if lang == "jpn":
        # Remove unwanted characters (hypothetical - might be too aggressive)
        text = re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]', '', text)
    
    return text

def filter_oov(text, vocab):
    """Filter out characters not in vocabulary"""
    return ''.join([c for c in text if c in vocab])

# Mock vocabulary - might not contain all Japanese characters
vocab = set(['あ', 'い', 'う', 'え', 'お'])  # Extremely limited vocabulary

# Test case that would trigger the bug
input_text = "こんにちは世界"  # "Hello world" in Japanese
lang = "jpn"

# Preprocessing pipeline
processed = preprocess_text(input_text, lang)
print(f"After preprocessing: {processed}")

filtered = filter_oov(processed, vocab)
print(f"After OOV filtering: {filtered}")  # Will be empty with this vocab

# Expected output would show empty string after filtering
# because most characters aren't in our mock vocabulary