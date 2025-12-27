# Step 2: Attempt to import `data_utils` from `keras.utils` (which will fail)
from keras.utils import data_utils

# This will raise the following error:
# ImportError: cannot import name 'data_utils' from 'keras.utils'

# Step 2: Now the import works
from keras.utils import data_utils
print("Import successful!")