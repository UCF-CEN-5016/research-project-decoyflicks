import tensorflow as tf
from official.vision.examples.starter.example_config import ExampleTask

def main():
    batch_size = 1
    predicted_boxes = tf.zeros((0, 4))  # Empty tensor for predicted boxes
    dummy_input = tf.ones((batch_size, 224, 224, 3))  # Dummy input tensor

    task = ExampleTask()
    model = task.build_model()

    try:
        task._generate_detections_v2_class_aware(predicted_boxes, dummy_input)
    except ValueError as e:
        assert "List argument 'values' to 'ConcatV2' Op with length 0 shorter than minimum length 2." in str(e)
        print("ValueError caught successfully.")

if __name__ == "__main__":
    main()