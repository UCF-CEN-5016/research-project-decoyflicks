%cd /content/models/research/delf/
!pip install -q delf

import os

# Move into the delf repository
os.chdir('/content/models/research/delf')

# Attempt to run extract_features.py
%cd /content/models/research/delf/delf/python/examples/
!python3 extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path list_images.txt \
  --output_dir data/oxford5k_features