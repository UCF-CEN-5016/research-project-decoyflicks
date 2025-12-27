import os
import sys
import subprocess

def add_path_to_env(path: str, env_var: str = "PYTHONPATH"):
    """
    Adds a given path to an environment variable if it's not already present.
    """
    current_paths = os.environ.get(env_var, "").split(os.pathsep)
    if path not in current_paths:
        if os.environ.get(env_var):
            os.environ[env_var] = f"{os.environ[env_var]}{os.pathsep}{path}"
        else:
            os.environ[env_var] = path
        print(f"Added '{path}' to {env_var}.")
    else:
        print(f"'{path}' already in {env_var}.")

def add_path_to_sys_path(path: str):
    """
    Inserts a given path to sys.path if it's not already present.
    """
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added '{path}' to sys.path.")
    else:
        print(f"'{path}' already in sys.path.")

def verify_and_import_transformers_deepspeed(transformers_base_path: str):
    """
    Sets up environment paths for transformers and attempts to import deepspeed.
    """
    add_path_to_env(transformers_base_path, "PYTHONPATH")
    add_path_to_sys_path(transformers_base_path)

    try:
        import transformers.deepspeed
        print("Successfully imported transformers.deepspeed.")
    except ImportError as e:
        print(f"Error importing transformers.deepspeed: {e}")
        print("Please ensure 'transformers' and 'deepspeed' libraries are correctly installed and accessible.")
        print("You might need to run: pip install transformers deepspeed")
    except Exception as e:
        print(f"An unexpected error occurred during import: {e}")

if __name__ == "__main__":
    # Define the base path for transformers.
    # This path should point to the directory containing the 'transformers' package,
    # relative to where this script is executed.
    # For example, if the transformers repo is cloned into a subdirectory named 'transformers_repo'
    # relative to this script, set it to 'transformers_repo'.
    transformers_repo_path = "." 
    print(f"Attempting to set up environment for transformers from: {transformers_repo_path}")
    verify_and_import_transformers_deepspeed(transformers_repo_path)
    print("Environment setup and import attempt complete.")
