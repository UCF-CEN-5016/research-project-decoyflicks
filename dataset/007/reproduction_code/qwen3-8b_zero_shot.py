import sys

# Create a test config file with content in Latin-1 encoding
with open('test.config', 'w', encoding='latin-1') as f:
    f.write('testé')

# Attempt to read the file using UTF-8 encoding, which will trigger the error
try:
    with open('test.config', 'r', encoding='utf-8') as f:
        content = f.read()
        print("Content read successfully:")
        print(content)
except UnicodeDecodeError as e:
    print("UnicodeDecodeError occurred:")
    print(f"Error: {e}")