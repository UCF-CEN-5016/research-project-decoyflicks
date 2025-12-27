import os
import sys

# Ensure the path is added only once to avoid redundancy if the script is run multiple times
transformers_path = "/path/to/transformers"
if transformers_path not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{transformers_path}" if os.environ.get("PYTHONPATH") else transformers_path
if transformers_path not in sys.path:
    sys.path.insert(0, transformers_path)

try:
    import transformers.deepspeed
except ImportError as e:
    print(f"Error: {e}")

print("Reproduction complete")
