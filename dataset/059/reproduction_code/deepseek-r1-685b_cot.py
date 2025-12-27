import keras_cv
from tensorflow import keras
import tensorflow as tf

# Minimal RetinaNet setup that may trigger similar issues
try:
    # Try with default backbone
    model = keras_cv.models.RetinaNet(
        classes=10,
        bounding_box_format="xywh",
        backbone="resnet50"
    )
    
    # Dummy data matching expected format
    images = tf.random.uniform((2, 512, 512, 3))
    labels = {
        "boxes": tf.random.uniform((2, 10, 4)),
        "classes": tf.random.uniform((2, 10), maxval=10, dtype=tf.int32)
    }
    
    # Compile and fit - likely to trigger error
    model.compile(
        optimizer="adam",
        classification_loss="focal",
        box_loss="smoothl1"
    )
    model.fit(images, labels, epochs=1)
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Likely cause: Version mismatch between keras-core/keras and keras-cv")
    print("Try: pip install keras-core --upgrade")
    print("Or: Use stable release instead of GitHub source")