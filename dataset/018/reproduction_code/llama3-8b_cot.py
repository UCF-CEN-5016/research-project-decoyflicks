import os

# Set up minimal environment
os.environ['PYTHONPATH'] = ''  # Ensure clean Python environment
os.system('pip --version')  # Verify pip is installed and functioning

try:
    # Execute the reproducing command
    os.system('python -m pip install --use-feature=2020-resolver .')
except Exception as e:
    print(f"Error: {e}")