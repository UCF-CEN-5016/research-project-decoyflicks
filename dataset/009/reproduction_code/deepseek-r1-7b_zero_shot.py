import tensorflow as tf

wrapped_model = tf.function(
    lambda inputs: {
        ' detections': (tf.zeros([1, 6], dtype=tf.float32),),
    },
    input_signature=[
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8),
    ],
)

def get_output(model, input_data):
    return model(input_data)['detections']

model = DetectionModel(wrapped_model.signatures['default'])

# Check if 'outputs' attribute exists
if hasattr(model, 'outputs'):
    outputs = model.outputs
else:
    # Get the output using the function method instead
    outputs = get_output(model, tf.zeros([1280, 1280, 3], dtype=tf.uint8))
    
print(outputs)