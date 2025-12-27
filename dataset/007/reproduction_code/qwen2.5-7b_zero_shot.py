# Create a test config file with content in Latin-1 encoding
with open('test.config', 'w', encoding='latin-1') as f:
    f.write('testé')

def read_file(file_path, encoding):
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
            print("Content read successfully:")
            print(content)
    except UnicodeDecodeError as e:
        print("UnicodeDecodeError occurred:")
        print(f"Error: {e}")

# Attempt to read the file using UTF-8 encoding, which will trigger the error
read_file('test.config', 'utf-8')