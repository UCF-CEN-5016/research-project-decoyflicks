import sys

def filter_oov(text, allowed):
    return ''.join([c for c in text if c in allowed])

allowed = {'a', 'b', 'c'}
input_text = 'defg'
filtered = filter_oov(input_text, allowed)
print(filtered)