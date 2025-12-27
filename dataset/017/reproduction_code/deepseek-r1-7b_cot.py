model_dir = pathlib.Path(model_dir)/"saved_model"
print(model_dir)
model = tf.saved_model.load_v2(str(model_dir))

model_file = model_name + '.tar.gz'
model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)

model_dir = pathlib.Path(model_dir)/"saved_model"

model.signatures["serving_default"]

model = tf.saved_model.load_v2(str(model_dir))

model = tf.saved_model.load(str(model_dir))

import tarfile
import os
import tensorflow as tf

# Step 1: Extract the tar.gz file
gz_file = "model.tar.gz"
extract_dir = "extracted_dir"

with tarfile.open(gz_file, 'r:gz') as f:
    f.extractall(path=extract_dir)

# Step 2: Define function to load TF1 model from saved model directory
def load_model(model_dir):
    # Verify that the saved model files exist in the expected structure
    if not os.path.exists(os.path.join(model_dir, 'saved_model.pb')):
        raise FileNotFoundError("Saved model file not found")
    
    try:
        # Use tf.saved_model.load for TF1 compatibility (changed from load_v2)
        loaded = tf.saved_model.load(str(model_dir))
        
        # If a specific signature is known to be present
        if 'serving_default' in loaded.signatures:
            model = loaded.signatures["serving_default"]
        else:
            raise ValueError("Serving default signature not found")
            
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Step 3: Correctly set the saved_model directory
saved_model_dir = os.path.join(extracted_dir, 'saved_model')

# Step 4: Load and test the model (assuming the model has a serving_default signature)
model = load_model(saved_model_dir)

if model is not None:
    # Example inference code that needs to be adapted based on actual model structure
    def show_inference(model, image):
        raise NotImplementedError("Inference implementation not provided")
    
    # Test with an example image (adjust as needed)
    test_image = tf.random.uniform([1, 256, 256, 3])
    results = model(test_image)
    print(results)
else:
    print("Failed to load the model.")