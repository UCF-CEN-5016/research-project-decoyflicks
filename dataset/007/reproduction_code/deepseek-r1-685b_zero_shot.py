import tempfile
import os

# Create a temporary file with non-UTF-8 content (Latin-1 encoded)
with tempfile.NamedTemporaryFile(delete=False, mode='wb') as f:
    f.write(b'This is a test file with non-UTF-8 character: \xe9')  # é in Latin-1
    file_path = f.name

# Try to read it as UTF-8 (reproduces the error)
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
finally:
    os.unlink(file_path)