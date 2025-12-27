# Create a file with invalid UTF-8 bytes and try to read it as UTF-8 text

# Step 1: Write invalid UTF-8 bytes to a file
with open("invalid_utf8.config", "wb") as f:
    # Write some valid UTF-8 bytes + invalid byte 0xe9 in the middle
    # For example: b"some valid text \xe9 with invalid utf8"
    f.write(b"some valid text \xe9 with invalid utf8\n")

# Step 2: Attempt to read the file as UTF-8 text
try:
    with open("invalid_utf8.config", "r", encoding="utf-8") as f:
        content = f.read()  # This should raise UnicodeDecodeError
except UnicodeDecodeError as e:
    print(f"Caught UnicodeDecodeError: {e}")