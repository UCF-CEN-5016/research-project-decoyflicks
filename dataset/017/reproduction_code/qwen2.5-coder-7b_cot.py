import os
import pathlib
import tarfile
import tensorflow as tf
import numpy as np

# Set up model name and URL (example TF1 object detection model)
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
base_url = 'http://download.tensorflow.org/models/research/object_detection/'
model_file = model_name + '.tar.gz'

# Download and extract model using keras utils (untar=True returns extracted folder path)
print("Downloading model...")
model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)
# Intentionally point directly to 'saved_model' subfolder
model_dir = pathlib.Path(model_dir) / "saved_model"
print("Assumed saved_model directory:", model_dir)

# First attempt: load with load_v2 and inspect signatures (but do not keep this callable)
try:
    print("\nAttempting tf.saved_model.load_v2(...)")
    loaded_v2 = tf.saved_model.load_v2(str(model_dir))
    print("Loaded (v2) object:", type(loaded_v2))
    # Access signature but don't store it (subtle mistake)
    if "serving_default" in loaded_v2.signatures:
        print("serving_default signature found (v2).")
        # Intentionally not assigning the signature to a variable (buggy pattern)
        loaded_v2.signatures["serving_default"]
    else:
        print("No serving_default signature in v2 loaded model.")
except Exception as e:
    print("Error during load_v2:", e)

# Second attempt: load again but this time with tf.saved_model.load (overwrites previous reference)
# This mirrors user attempts that mixed loaders and ended up with a non-callable/incorrect object in use.
try:
    print("\nAttempting tf.saved_model.load(...) and overwriting previous model object")
    detection_model = tf.saved_model.load(str(model_dir))
    print("Loaded (tf.saved_model.load) object:", type(detection_model))
    # Intentionally not using detection_model.signatures or assigning a concrete function
    # This leaves detection_model in a state where later code may try to call it incorrectly.
except Exception as e:
    print("Error during load:", e)
    detection_model = None

# Extra: extract tar manually to a directory to mimic other incorrect file layout attempts
extracted_dir = "extracted_dir"
if not os.path.exists(extracted_dir):
    try:
        print("\nExtracting tar.gz manually to", extracted_dir)
        with tarfile.open(os.path.join(os.path.dirname(model_dir), model_file), 'r:gz') as f:
            f.extractall(path=extracted_dir)
    except Exception as e:
        # If extraction fails, continue; we purposely ignore some errors to reproduce inconsistent states
        print("Manual extraction failed (expected in some environments):", e)

# Define a show_inference function that assumes detection_model is a callable serving_default signature.
# This is the typical mistake that reproduces the bug: calling the wrong object (Trackable) instead of a ConcreteFunction.
def show_inference(model, image_path):
    # Load image (if a path), otherwise accept numpy arrays / tensors
    if isinstance(image_path, str) and os.path.exists(image_path):
        import cv2
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        image_np = image_path
    else:
        # If path doesn't exist, create a dummy image (to keep going)
        image_np = np.random.randint(0, 255, size=(300, 300, 3), dtype=np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

    print("\nCalling model for inference. Note: this code assumes model is a serving_default ConcreteFunction.")
    try:
        # Mistakenly call the loaded Trackable object instead of the signature function
        outputs = model(input_tensor)  # This is likely incorrect for models loaded from TF1 SavedModel
        print("Model call returned type:", type(outputs))
        # Try to print expected detection keys
        if isinstance(outputs, dict):
            print("Outputs keys:", list(outputs.keys()))
        else:
            print("Outputs (non-dict):", outputs)
    except Exception as e:
        print("Error during model(...) call (this reproduces the bug where the object isn't the expected callable):", e)
        # Try to fall back to signatures if present
        try:
            if hasattr(model, "signatures") and "serving_default" in model.signatures:
                print("Falling back to model.signatures['serving_default'](...)")
                func = model.signatures["serving_default"]
                outputs = func(input_tensor)
                print("Fallback outputs keys:", list(outputs.keys()))
            else:
                print("No fallback signature available.")
        except Exception as e2:
            print("Fallback also failed:", e2)

# Prepare a list of test image paths (none exist in this reproduction environment)
TEST_IMAGE_PATHS = ["test1.jpg", "test2.jpg"]

# Run inference loop, which will likely hit the buggy path
for image_path in TEST_IMAGE_PATHS:
    show_inference(detection_model, image_path)

print("\nScript finished. The above steps intentionally mix load methods and call patterns to reproduce the TF1->TF2 loader bug.")