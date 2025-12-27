import tensorflow as tf

# Attempt to read a file with invalid UTF-8 encoding
def read_file(file_path):
    try:
        with tf.io.gfile.GFile(file_path, 'r') as f:
            contents = f.read()
            return contents
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")

# Create a file with invalid UTF-8 encoding
with open('example.txt', 'wb') as f:
    f.write(b'\xe9')  # Write a single byte that's not valid UTF-8

# Try to read the file
read_file('example.txt')