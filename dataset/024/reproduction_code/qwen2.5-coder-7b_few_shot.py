import sys
import importlib

DEFAULT_DELF_PYTHON_PATH = "/content/models/research/delf/delf/python"


def _append_delf_path(path: str) -> None:
    """Append the DELF python path to sys.path (unconditionally, as originally)."""
    sys.path.append(path)


def load_google_landmarks_module() -> None:
    """
    Attempt to import the Google Landmarks dataset module from the DELF package.
    Prints 'Import succeeded' on success and 'Import failed: <error>' on ModuleNotFoundError.
    """
    _append_delf_path(DEFAULT_DELF_PYTHON_PATH)
    try:
        importlib.import_module("delf.python.datasets.google_landmarks_dataset.googlelandmarks")
        print("Import succeeded")
    except ModuleNotFoundError as e:
        print(f"Import failed: {e}")


if __name__ == "__main__":
    load_google_landmarks_module()