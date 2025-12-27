import tempfile
from tensorflow.python.lib.io import file_io

# Create a temp file with non-UTF-8 content
with tempfile.NamedTemporaryFile(delete=False) as f:
    file_path = f.name
    f.write(b'Some text with non-UTF8 char: \xe9')  # Latin-1 character

# Attempt to read with UTF-8 encoding (will fail)
try:
    with file_io.FileIO(file_path, mode='r') as f:  # Default is UTF-8
        content = f.read()
except UnicodeDecodeError as e:
    print(f"Error: {e}")
    print("This reproduces the config file reading issue")

# Clean up
import os
os.unlink(file_path)