# Necessary imports
import importlib.util
import sys

# Minimal environment setup
try:
    import tensorflow as tf
except ImportError:
    print("Installing tensorflow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf

try:
    import tensorflow_text as tf_text
except ImportError:
    print("Installing tensorflow-text...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-text"])
    import tensorflow_text as tf_text

# Triggering conditions
try:
    import tensorflow_models as tfm
except ImportError as e:
    print(f"Error importing tensorflow_models: {e}")

# Verify the presence of the tensorflow_text package
spec = importlib.util.find_spec('tensorflow_text')
if spec is not None:
    print("tensorflow_text package found.")
else:
    print("tensorflow_text package not found.")