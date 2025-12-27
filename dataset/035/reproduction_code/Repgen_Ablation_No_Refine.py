import torch

# Clone the repository https://github.com/labmlai/annotated_deep_learning_paper_implementations to a local directory
# Navigate to the directory labml_nn/transformers/rope/
# Open and edit the file __init__.py
# Locate line 226 and replace `rotary_pe = RotaryPositionalEmbeddings(3)` with `rotary_pe = RotaryPositionalEmbeddings(4)`
# Save the changes to __init__.py

# Navigate back to the root directory of the repository
# Install dependencies using pip: `pip install -r requirements.txt`
# Set up a Python environment and activate it (e.g., using venv)

# Run the performance_test.py script from labml_nn/optimizers/performance_test.py
# Observe the output in the console to verify if an error occurs at line 226 of __init__.py

# If no error occurs, re-run the script multiple times and monitor for any signs of incorrect behavior or errors related to the RotaryPositionalEmbeddings initialization

# After confirming the bug is reproduced, revert the change in __init__.py back to `rotary_pe = RotaryPositionalEmbeddings(3)` to resolve the issue