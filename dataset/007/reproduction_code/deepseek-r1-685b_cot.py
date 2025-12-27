import tensorflow as tf
from object_detection.utils import config_util

# Reproduction of the error (assuming non-UTF-8 config file)
try:
    # This will fail if config file has non-UTF-8 encoding
    configs = config_util.get_configs_from_pipeline_file(
        "ssd_efficientdet_d0_512x512_coco17_tpu-8.config"
    )
except UnicodeDecodeError as e:
    print(f"Original error: {e}")

# Solution 1: Explicitly specify encoding (try common Windows encodings)
try:
    configs = config_util.get_configs_from_pipeline_file(
        "ssd_efficientdet_d0_512x512_coco17_tpu-8.config",
        encoding='cp1252'  # Common Windows encoding
    )
    print("Successfully loaded config with cp1252 encoding")
except Exception as e:
    print(f"Failed with cp1252: {e}")

# Solution 2: Convert the config file to UTF-8
def convert_config_to_utf8(input_path, output_path):
    with open(input_path, 'rb') as f:
        content = f.read()
    try:
        # Try common encodings until one works
        for encoding in ['cp1252', 'latin1', 'iso-8859-1']:
            try:
                decoded = content.decode(encoding)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(decoded)
                print(f"Successfully converted using {encoding}")
                return True
            except UnicodeDecodeError:
                continue
        return False
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

if convert_config_to_utf8(
    "ssd_efficientdet_d0_512x512_coco17_tpu-8.config",
    "ssd_efficientdet_d0_512x512_coco17_tpu-8_utf8.config"
):
    configs = config_util.get_configs_from_pipeline_file(
        "ssd_efficientdet_d0_512x512_coco17_tpu-8_utf8.config"
    )
    print("Successfully loaded converted config file")