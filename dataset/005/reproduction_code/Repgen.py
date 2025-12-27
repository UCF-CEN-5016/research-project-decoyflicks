import tensorflow as tf

try:
    # This is the problematic import mentioned in the bug report
    from tensorflow.python.framework import tensor
    print("Import successful (unexpected)")
except ImportError as e:
    print(f"Bug reproduced - ImportError: {e}")
    
    # Print tensorflow version for reference
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for related modules to help diagnose the issue
    try:
        from tensorflow.python.framework import ops
        print("Note: 'ops' module can be imported from tensorflow.python.framework")
    except ImportError:
        print("Note: 'ops' module also cannot be imported")
        
    try:
        # In newer TF versions, this may be the correct import path
        from tensorflow.python.framework.tensor import Tensor
        print("Alternative import path works")
    except ImportError:
        print("Alternative import path also fails")

if __name__ == "__main__":
    # The bug is in the import system, so no need to actually run model code
    pass