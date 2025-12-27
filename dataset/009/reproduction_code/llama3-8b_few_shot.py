import tensorflow as tf
from object_detection import exporter_main_v2, model_lib_v2

# Train a model
model_config = model_lib_v2.get_initial_model_config()
train_config = model_lib_v2.get_initial_train_config()
hyperparameters = {}
train_config.merge_from_dict(hyperparameters)

 DetectionFromImageModule = model_lib_v2.DetectionFromImageModule(
    config=model_config, train_config=train_config)

# Try to export the model
exporter_main_v2.export_inference_graph(DetectionFromImageModule)