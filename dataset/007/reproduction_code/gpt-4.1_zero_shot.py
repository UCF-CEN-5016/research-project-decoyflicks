with open("corrupt_utf8.txt", "wb") as f:
    f.write(b"valid ascii text " + b"\xe9" + b" more text")

with open("corrupt_utf8.txt", "r", encoding="utf-8") as f:
    f.read()