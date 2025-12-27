import tensorflow as tf

# Simulate a scenario where the input list is empty (e.g., no predicted boxes)
values = []

if len(values) < 2:
    # Handle empty or insufficient input (e.g., return a zero tensor)
    result = tf.zeros(shape=[0, 4], dtype=tf.float32)
else:
    try:
        result = tf.concat(values, axis=0)
        print("Concatenation executed successfully.")
    except ValueError as e:
        print(f"Error occurred during concatenation: {e}")

print("Result shape:", result.shape)