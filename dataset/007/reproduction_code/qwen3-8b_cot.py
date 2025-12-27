# Reproduction of UnicodeDecodeError
try:
    with open("invalid_encoding_file.txt", "r", encoding="utf-8") as f:
        content = f.read()
    print(content)
except UnicodeDecodeError as e:
    print(f"Error: {e}")

with open("file.txt", "r", encoding="utf-8-sig") as f:
      content = f.read()

# Test with a UTF-8 file
try:
    with open("valid_utf8_file.txt", "r", encoding="utf-8") as f:
        content = f.read()
    print("File read successfully!")
except UnicodeDecodeError as e:
    print(f"Error: {e}")