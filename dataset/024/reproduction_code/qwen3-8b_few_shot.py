import sys
sys.path.append('/content/models/research/delf/delf/python')  # Simulate directory structure

# Attempt to import from a directory missing __init__.py
try:
    from delf.python.datasets.google_landmarks_dataset import googlelandmarks as gld
    print("Import succeeded")
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")