import tensorflow as tf
from object_detection.builders import model_builder

# Create a simple model builder
model_config = {
    'model': {
        'ssd': {
            'num_classes': 1,
            'image_resizer': {
                'fixed_shape_resizer': {
                    'height': 512,
                    'width': 512
                }
            }
        }
    }
}

# Try to build the model
model = model_builder.build(model_config, is_training=True)

print(model.summary())