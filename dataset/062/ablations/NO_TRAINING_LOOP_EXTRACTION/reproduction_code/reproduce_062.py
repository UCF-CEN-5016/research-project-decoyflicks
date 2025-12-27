import pandas as pd
import numpy as np
import sys
import tensorflow as tf

# Create DataFrame 'train_df' with 100 rows and 3 columns
train_df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=['A', 'B', 'C'])

# Create another DataFrame 'test_df' with 50 rows and 3 columns
test_df = pd.DataFrame(np.random.randint(0, 100, size=(50, 3)), columns=['A', 'B', 'C'])

# Attempt to concatenate 'train_df' and 'test_df' using the deprecated method
all_data = train_df.append(test_df)

# Print versions
print("Pandas version:", pd.__version__)
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)