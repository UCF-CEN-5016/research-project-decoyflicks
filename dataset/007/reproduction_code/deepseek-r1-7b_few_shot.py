import sys

# Example showing UnicodeDecodeError due to invalid continuation byte
def main():
    # Create a file with invalid UTF-8 sequence (byte 0xe9)
    with open('invalid_utf8.txt', 'wb') as f:
        f.write(b'\xe9')  # Single invalid byte
    
    # Or try to decode without specifying encoding
    with open('test.txt', 'r') as f:
        print(f.read().replace('\n', '\r\n'))  # Replace newlines for Windows compatibility

if __name__ == "__main__":
    main()