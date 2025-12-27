def read_file(filename, encoding):
    try:
        with open(filename, "r", encoding=encoding) as f:
            content = f.read()
            print(content)
    except UnicodeDecodeError as e:
        print(f"Error reading file {filename}: {e}")

# Reproduction of UnicodeDecodeError
read_file("invalid_encoding_file.txt", "utf-8")

# Reading a file with utf-8-sig encoding
read_file("file.txt", "utf-8-sig")

# Test with a UTF-8 file
read_file("valid_utf8_file.txt", "utf-8")