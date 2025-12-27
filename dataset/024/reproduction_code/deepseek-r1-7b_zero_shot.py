# Example code ensuring delf is a proper Python package

import os
from delf.python.datasets import google_landmarks_dataset as gld

if __name__ == "__main__":
    print(f"Google Landmarks Dataset loaded: {gld}")