import os
import tensorflow as tf
import data_utils
import helper_utils
import numpy as np

# Create directory structure
os.makedirs('models/centernet_mobilenetv2fp', exist_ok=True)

# Create a dummy pipeline.config file
with open('models/centernet_mobilenetv2fp/pipeline.config', 'w') as f:
    f.write('model: {name: "non_existent_model"}\n')

# Dummy model_main_tf2.py script
def main():
    from object_detection import exporter  # This will trigger the import error
    exporter.export_inference_graph()

if __name__ == '__main__':
    main()