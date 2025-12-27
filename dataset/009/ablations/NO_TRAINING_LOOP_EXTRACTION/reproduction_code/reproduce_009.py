import os
import tempfile
import tensorflow as tf
from orbit import actions

class MockDetectionFromImageModule(tf.Module):
    def __init__(self):
        self.outputs = None  # Simulating the absence of 'outputs' attribute

    @tf.function(input_signature=[])
    def __call__(self):
        return tf.constant(0)

def main():
    model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
    with tempfile.TemporaryDirectory() as temp_dir:
        base_name = os.path.join(temp_dir, 'model_export')
        file_manager = actions.ExportFileManager(base_name, max_to_keep=2)
        model = MockDetectionFromImageModule()
        export_action = actions.ExportSavedModel(model, file_manager=file_manager, signatures=model.__call__)

        try:
            export_action({})
            _ = model.outputs  # Attempt to access 'outputs' attribute
        except AttributeError as e:
            print(e)

if __name__ == '__main__':
    main()