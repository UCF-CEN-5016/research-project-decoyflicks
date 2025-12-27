# First install required packages
!pip install -q tensorflow==2.12.0
!pip install -q git+https://github.com/tensorflow/models.git

# Now try the problematic imports
try:
    from object_detection.utils import config_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    print("Trying alternative solution...")
    
    # Common fix - add models/research to path
    import sys
    import os
    from pathlib import Path
    
    # Adjust this path if needed
    models_dir = Path('models/research')
    if not models_dir.exists():
        !git clone --depth 1 https://github.com/tensorflow/models
    sys.path.append(str(models_dir))
    
    # Install protobuf compiler and compile protos
    !apt-get install -qq protobuf-compiler
    !cd models/research && protoc object_detection/protos/*.proto --python_out=.
    
    # Try imports again
    try:
        from object_detection.utils import config_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        print("Imports successful after setup!")
    except ImportError as e:
        print(f"Still failing: {e}")