#!/usr/bin/env python3

import os
import subprocess

def main():
    # Navigate to the examples directory
    os.chdir('/content/models/research/delf/delf/python/examples/')
    print(f"Current directory: {os.getcwd()}")
    
    # Check if required files exist
    print("Checking for required files...")
    required_files = ["extract_features.py", "delf_config_example.pbtxt", "list_images.txt"]
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            
    # Create output directory if it doesn't exist
    os.makedirs("data/oxford5k_features", exist_ok=True)
    print("✓ Created output directory: data/oxford5k_features")
    
    # Run the problematic command
    print("\nRunning command to reproduce error:")
    print("python3 extract_features.py --config_path delf_config_example.pbtxt --list_images_path list_images.txt --output_dir data/oxford5k_features")
    
    try:
        result = subprocess.run([
            "python3", "extract_features.py",
            "--config_path", "delf_config_example.pbtxt",
            "--list_images_path", "list_images.txt",
            "--output_dir", "data/oxford5k_features"
        ], capture_output=True, text=True)
        
        # Display output
        if result.stdout:
            print("\nCommand output:")
            print(result.stdout)
        
        # Display error (this should contain the ModuleNotFoundError)
        if result.stderr:
            print("\nError output:")
            print(result.stderr)
            
        # Check if the expected error is in the output
        if "No module named 'delf.python.datasets'" in result.stderr:
            print("\n✓ Successfully reproduced the bug: ModuleNotFoundError: No module named 'delf.python.datasets'")
        else:
            print("\n✗ Did not reproduce the expected error")
            
    except Exception as e:
        print(f"\nException running command: {e}")

if __name__ == "__main__":
    main()