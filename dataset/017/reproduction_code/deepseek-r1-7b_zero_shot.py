import os
from tensorflow.keras.utils import get_file

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    saved_model_path = os.path.join(model_dir, "saved_model")
    
    print(saved_model_path)
    
    # Load the saved model using TF 2.x compatible methods
    model = tf.saved_model.load(
        saved_model_path,
        signature='serving_default',
        tags={'serving': ['default']})
    
    return model

# Example usage:
model_name = 'faster_rcnn_inception_v2_coco_117600'
test_image_paths = [os.path.join('data', 'person', f'{i}.png') for i in range(1, 4)]
detection_model = load_model(model_name)

for image_path in test_image_paths:
    show_inference(detection_model, image_path)