import tensorflow as tf

# Create a file with non-UTF-8 content (Latin-1 encoding)
with open("pipeline.config", "w", encoding="latin-1") as f:
    f.write("Some content with é character")

# Attempt to read the file using TensorFlow's file IO
try:
    with tf.io.gfile.GFile("pipeline.config", "r") as f:
        content = f.read()
        print("Successfully read file:", content)
except UnicodeDecodeError as e:
    print("Caught UnicodeDecodeError:", e)