import sys

# Simulate directory structure
sys.path.append('/content/models/research/delf/delf/python')

def import_google_landmarks_dataset():
    try:
        from delf.python.datasets.google_landmarks_dataset import googlelandmarks as gld
        print("Import succeeded")
    except ModuleNotFoundError as e:
        print(f"Import failed: {e}")

if __name__ == "__main__":
    import_google_landmarks_dataset()