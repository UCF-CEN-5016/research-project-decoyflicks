# Step 1: Import `data_utils` from `keras.utils` (which will fail)
try:
    from keras.utils import data_utils
except ImportError:
    print("Import failed")

# Step 2: Now the import works
from keras.utils import data_utils
print("Import successful!")