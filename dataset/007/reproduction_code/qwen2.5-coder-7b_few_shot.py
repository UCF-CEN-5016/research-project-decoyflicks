import tensorflow as tf
from typing import Optional


def write_latin1_file(path: str, content: str) -> None:
    """Write text to a file using Latin-1 encoding."""
    with open(path, "w", encoding="latin-1") as fh:
        fh.write(content)


def read_with_tensorflow(path: str) -> Optional[str]:
    """Read a file using TensorFlow's file IO and return its content.

    If a UnicodeDecodeError occurs during read, it is caught and logged,
    and None is returned.
    """
    try:
        with tf.io.gfile.GFile(path, "r") as fh:
            return fh.read()
    except UnicodeDecodeError as e:
        print("Caught UnicodeDecodeError:", e)
        return None


def main() -> None:
    file_path = "pipeline.config"
    write_latin1_file(file_path, "Some content with é character")

    content = read_with_tensorflow(file_path)
    if content is not None:
        print("Successfully read file:", content)


if __name__ == "__main__":
    main()