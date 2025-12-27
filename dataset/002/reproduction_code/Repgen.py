import tensorflow as tf
import sys

def reproduce_bug():
    print(f"Using TensorFlow version {tf.__version__}")
    
    try:
        # Import the exporter module, which tries to import tensorflow.contrib
        from object_detection.exporter import exporter
        
        # We can try to access any function that will use the contrib modules
        # For example, exporter.export_inference_graph would eventually use
        # tensorflow.contrib.tfprof and tensorflow.contrib.quantize.python.graph_matcher
        
        print("Successfully imported exporter module (this might happen, but functions will fail)")
        
        # Try to call a function that will use the contrib modules
        # This is similar to how model_main_tf2.py would use the exporter
        pipeline_config_file = 'models/centernet_mobilenetv2fp/pipeline.config'
        try:
            # This will fail when trying to profile the graph using contrib.tfprof
            exporter.export_inference_graph(
                'image_tensor',
                pipeline_config_file,
                'models/centernet_mobilenetv2fp/model.ckpt',
                'output_dir'
            )
        except Exception as e:
            if isinstance(e, ModuleNotFoundError) and 'tensorflow.contrib' in str(e):
                print(f"Successfully reproduced the error during function call: {e}")
                return e
            else:
                print(f"Different error occurred: {e}")
                return e
                
    except ModuleNotFoundError as e:
        if 'tensorflow.contrib' in str(e):
            print(f"Successfully reproduced the error during import: {e}")
            return e
        else:
            print(f"Different ModuleNotFoundError: {e}")
            return e
    
    return None

if __name__ == "__main__":
    error = reproduce_bug()
    
    if error and 'tensorflow.contrib' in str(error):
        print("\n✓ Successfully reproduced the bug from the command line execution")
        print("\nThis matches the error you would see when running:")
        print("python model_main_tf2.py --model_dir=models/centernet_mobilenetv2fp --pipeline_config_path=models/centernet_mobilenetv2fp/pipeline.config")
        
        print("\nTo fix this issue, consider one of the following solutions:")
        print("1. Use the TensorFlow 2.x compatible version of the Object Detection API")
        print("   - Instead of using exporter.py, use exporter_lib_v2.py from object_detection/exporters")
        print("2. Downgrade to TensorFlow 1.x (not recommended)")
        print("3. For training with TF2, use the model_main_tf2.py script from the TF2 Object Detection API")