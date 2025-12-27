import os

# Set environment variables
os.environ['HF_PATH'] = '../'
os.environ['NGPUS'] = '6'

# Run test_tune.sh
os.system('./test_tune.sh tune')