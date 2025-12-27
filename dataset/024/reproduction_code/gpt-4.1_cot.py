import os
import sys
import tempfile
import shutil

# Step 1: Setup minimal delf package structure in a temp directory
temp_dir = tempfile.mkdtemp()
print(f"Created temp directory at {temp_dir}")

try:
    # Paths for packages
    delf_dir = os.path.join(temp_dir, 'delf')
    python_dir = os.path.join(delf_dir, 'python')
    datasets_dir = os.path.join(python_dir, 'datasets')
    training_dir = os.path.join(python_dir, 'training')
    model_dir = os.path.join(training_dir, 'model')

    # Create directories
    os.makedirs(datasets_dir)
    os.makedirs(model_dir)

    # Create __init__.py files except for datasets (simulate missing __init__.py)
    open(os.path.join(delf_dir, '__init__.py'), 'w').close()
    open(os.path.join(python_dir, '__init__.py'), 'w').close()
    # datasets_dir: NO __init__.py to trigger the bug

    open(os.path.join(training_dir, '__init__.py'), 'w').close()
    open(os.path.join(model_dir, '__init__.py'), 'w').close()

    # Create google_landmarks_dataset.py under datasets
    with open(os.path.join(datasets_dir, 'google_landmarks_dataset.py'), 'w') as f:
        f.write("""
def dummy_func():
    return "Hello from google_landmarks_dataset"
""")

    # Create export_model_utils.py under model that imports datasets module
    with open(os.path.join(model_dir, 'export_model_utils.py'), 'w') as f:
        f.write("""
# This import should fail if datasets is not a package
from delf.python.datasets.google_landmarks_dataset import dummy_func

def call_dummy():
    return dummy_func()
""")

    # Add the temp_dir to sys.path
    sys.path.insert(0, temp_dir)

    # Step 2: Try importing export_model_utils and call function
    print("Trying to import delf.python.training.model.export_model_utils with missing datasets/__init__.py ...")
    try:
        import delf.python.training.model.export_model_utils as export_model_utils
        result = export_model_utils.call_dummy()
        print("Unexpectedly succeeded:", result)
    except ModuleNotFoundError as e:
        print("Expected failure:", e)

    # Step 3: Fix by adding __init__.py in datasets directory
    with open(os.path.join(datasets_dir, '__init__.py'), 'w') as f:
        f.write("# Make datasets a package\n")

    # Now try import again
    print("\nAdded __init__.py to datasets/, retrying import...")
    # Remove cached modules to force re-import
    modules_to_remove = [k for k in sys.modules if k.startswith('delf')]
    for k in modules_to_remove:
        del sys.modules[k]

    import delf.python.training.model.export_model_utils as export_model_utils
    result = export_model_utils.call_dummy()
    print("Succeeded after fix:", result)

finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    sys.path.remove(temp_dir)