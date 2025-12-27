# Simulate reading a file containing bytes invalid in UTF-8 encoding
file_path = 'example_config.config'

# Write bytes that are invalid UTF-8 (0xe9 alone is invalid)
with open(file_path, 'wb') as f:
    f.write(b"valid_line=1\ninvalid_char=\xe9\nanother_line=2\n")

# Attempt to read the file using the default UTF-8 encoding (raises UnicodeDecodeError)
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)