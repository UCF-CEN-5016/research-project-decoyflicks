# Import the required modules
import sys

# Attempt to import transformers.deepspeed
try:
    from transformers import deepspeed
except ImportError as e:
    print(f"Error importing transformers.deepspeed: {e}")
    # If the import fails, print the error and exit
    sys.exit(1)

# If the import succeeds, this line should be reached without errors
print("Successfully imported transformers.deepspeed")