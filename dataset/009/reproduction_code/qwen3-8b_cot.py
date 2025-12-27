# 1. Necessary imports
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.modeling import model_builder
from object_detection.protos import model_pb2
from object_detection.exporter_main_v2 import main as exporter_main
from object_detection.model_main_tf2 import main as train_main

# 2. Set up environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 3. Define paths and configurations
config_path = 'configs/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config'
train_data_path = 'path/to/train.record'
val_data_path = 'path/to/val.record'
output_dir = 'output_model'
export_dir = 'exported_model'

# 4. Train the model (simplified)
def train_model():
    # Load and parse config
    config = config_util.get_configs_from_pipeline_file(config_path)
    config = config_util.merge_from_string(config, 'train_config.max_num_steps=10000')
    
    # Build model
    model = model_builder.build(model_config=config['model'], is_training=True)
    
    # Train loop (simplified)
    for step in range(10000):
        # Simulate training
        pass

    # Save model
    model.save_pretrained(output_dir)

# 5. Export the model (triggering the bug)
def export_model():
    # Load config and model
    config = config_util.get_configs_from_pipeline_file(config_path)
    model = model_builder.build(model_config=config['model'], is_training=False)
    
    # Export to SavedModel
    exporter_main(
        pipeline_config_path=config_path,
        model_dir=output_dir,
        output_dir=export_dir,
        use_tpu=False
    )

# 6. Execute steps
if __name__ == '__main__':
    train_model()
    export_model()