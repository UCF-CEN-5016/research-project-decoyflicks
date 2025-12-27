import os
import subprocess

# Step 1: Clone the delf repository
subprocess.run(["git", "clone", "https://github.com/mirzaim/models.git"])

# Step 2: Change the working directory
os.chdir("/content/models/research/delf/delf/python/examples/")

# Step 3: Download the example configuration file and image list
with open("delf_config_example.pbtxt", "w") as f:
    f.write("your_config_content_here")  # Replace with actual config content

with open("list_images.txt", "w") as f:
    f.write("/content/image1.jpg\n")
    f.write("/content/image2.jpg\n")
    f.write("/content/image3.jpg\n")
    f.write("/content/image4.jpg\n")
    f.write("/content/image5.jpg\n")

# Step 4: Run the feature extraction script
subprocess.run(["python3", "extract_features.py", 
                "--config_path", "delf_config_example.pbtxt", 
                "--list_images_path", "list_images.txt", 
                "--output_dir", "data/oxford5k_features"])

# Step 5: Check for the error
if not os.path.exists("delf/python/datasets/__init__.py"):
    print("ModuleNotFoundError: No module named 'delf.python.datasets'")