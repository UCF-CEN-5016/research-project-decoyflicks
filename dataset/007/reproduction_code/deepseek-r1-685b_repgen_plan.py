import tempfile
import os
from tensorflow.python.lib.io import file_io

def create_temp_file_with_non_utf8_content():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name
        temp_file.write(b'Some text with non-UTF8 char: \xe9')  # Latin-1 character
    return file_path

def read_file_content(file_path):
    try:
        with file_io.FileIO(file_path, mode='r') as file:
            content = file.read()
    except UnicodeDecodeError as e:
        print(f"Error: {e}")
        print("This reproduces the config file reading issue")

def cleanup_temp_file(file_path):
    os.unlink(file_path)

file_path = create_temp_file_with_non_utf8_content()
read_file_content(file_path)
cleanup_temp_file(file_path)