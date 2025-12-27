# This script checks if the commons module exists in the Python path
import sys
import os
import importlib.util

def check_module_exists(module_name):
    """Check if a module exists in the Python path"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def main():
    # Check if commons module exists
    commons_exists = check_module_exists("commons")
    print(f"Commons module exists: {commons_exists}")
    
    # If we're in a fairseq repo, check the examples/mms/tts directory structure
    if os.path.exists("examples/mms/tts"):
        print("Found examples/mms/tts directory")
        
        # Check if infer.py imports commons module
        infer_path = "examples/mms/tts/infer.py"
        if os.path.exists(infer_path):
            with open(infer_path, "r") as f:
                content = f.read()
                if "import commons" in content or "from commons import" in content:
                    print("infer.py imports commons module")
                else:
                    print("infer.py does not import commons module directly")
    
if __name__ == "__main__":
    main()