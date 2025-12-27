```python
%cd /content/models/research/delf/delf/python/examples/
!mkdir datasets
!touch datasets/__init__.py
!python3 extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path list_images.txt \
  --output_dir data/oxford5k_features

