import os
import subprocess

# Step 1: Clone the TensorFlow Models repository (if not already cloned)
if not os.path.exists("models"):
    subprocess.run(["git", "clone", "https://github.com/tensorflow/models.git"])

# Step 2: Navigate to the research directory
os.chdir("models/research")

# Step 3: Set up the environment variables (required for imports)
os.environ['PYTHONPATH'] += os.pathsep + os.path.abspath(".")

# Step 4: Install required dependencies
!pip install -q tensorflow==2.12 Cython pillow lxml

# Step 5: Compile protos (if needed)
subprocess.run(["python", "setup.py", "build", "--build_dir=object_detection/protos"])
subprocess.run(["python", "setup.py", "build", "--build_dir=ops"])

# Step 6: Import required libraries (corrected import paths)
try:
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils
    from object_detection.utils import config_util
    from object_detection.builders import model_builder
    from object_detection.protos import pipeline_pb2
    print("Imports successful!")
except ImportError as e:
    print(f"ImportError: {e}")